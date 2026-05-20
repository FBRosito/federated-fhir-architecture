# Decisões de Arquitetura — FL-FHIR Architecture

**Projeto:** Aprendizado Federado sobre Dados Clínicos no Padrão FHIR
**Repositório:** `federated-fhir-architecture`
**Versão do documento:** 3.0
**Data:** 2026-05-20

---

## Sumário

1. [FedProx + Privacidade Diferencial Client-Side](#1-fedprox--privacidade-diferencial-client-side)
   - 1.1 [O problema: dados Non-IID em ambientes hospitalares](#11-o-problema-dados-non-iid-em-ambientes-hospitalares)
   - 1.2 [Por que FedProx e não FedAvg](#12-por-que-fedprox-e-não-fedavg)
   - 1.3 [Por que DP-SGD client-side (Opacus)](#13-por-que-dp-sgd-client-side-opacus)
   - 1.4 [Modelo de ameaça mitigado](#14-modelo-de-ameaça-mitigado)
   - 1.5 [Limitações conhecidas](#15-limitações-conhecidas)
2. [Llama-3 quantizado em 4-bits + LoRA](#2-llama-3-quantizado-em-4-bits--lora)
   - 2.1 [A restrição de hardware como requisito de projeto](#21-a-restrição-de-hardware-como-requisito-de-projeto)
   - 2.2 [Quantização NF4 via BitsAndBytes](#22-quantização-nf4-via-bitsandbytes)
   - 2.3 [Por que LoRA e não fine-tuning completo](#23-por-que-lora-e-não-fine-tuning-completo)
   - 2.4 [Configuração dos adaptadores LoRA](#24-configuração-dos-adaptadores-lora)
   - 2.5 [Eficiência de comunicação no contexto federado](#25-eficiência-de-comunicação-no-contexto-federado)
3. [Referências](#3-referências)

---

## 1. FedProx + Privacidade Diferencial Client-Side

### 1.1 O problema: dados Non-IID em ambientes hospitalares

Dados clínicos reais não seguem distribuições i.i.d. entre instituições de saúde. O perfil diagnóstico de um hospital cardiológico é radicalmente diferente do de um centro de pneumologia ou de uma UPA de atendimento geral. No presente sistema, esse fenômeno é modelado explicitamente pelo `etl_worker`, que distribui os dados por especialidade médica:

| Partição | Perfil da Unidade | Predominância diagnóstica | Códigos CID-10 típicos |
|---|---|---|---|
| 0 | Hospital A — Cardiologia | Cardiovascular (70%) | I10, I50.0, I20.0, I63.9 |
| 1 | Hospital B — Pneumologia | Respiratório (70%) | J44.1, J45.9, J18.9, I26.9 |
| 2 | Centro C — Endocrinologia | Metabólico/Endócrino (70%) | E11.9, E03.9, E28.2, E21.0 |
| 3 | UPA D — Geral | Distribuição uniforme | M54.5, N39.0, F32.9, N18.3 |

Essa heterogeneidade não é um artefato de simulação — ela reflete a realidade de qualquer rede de saúde e é o principal desafio técnico do Aprendizado Federado aplicado à área médica.

### 1.2 Por que FedProx e não FedAvg

O **FedAvg** (McMahan et al., 2017) é o algoritmo de referência para FL. Ele assume que os gradientes dos clientes convergem, em média, para o gradiente do objetivo global. Essa premissa vale para dados i.i.d., mas **falha em distribuições Non-IID**: cada cliente otimiza uma função de perda local diferente, e a média simples dos gradientes pode apontar em direções conflitantes — fenômeno conhecido como *client drift*.

O **FedProx** (Li et al., 2020b) resolve isso adicionando um **termo proximal** à função de perda local de cada cliente:

```
L_FedProx(w) = L_local(w) + (μ/2) · ‖w − w_global‖²
```

O efeito prático é que cada cliente é penalizado por se desviar excessivamente do modelo global. O hiperparâmetro μ controla a intensidade dessa restrição:

- μ → 0: degrada para FedAvg (sem restrição)
- μ → ∞: clientes não atualizam (modelo global estático)
- μ = 0.01 (valor adotado): permite adaptação local com estabilidade de convergência

A escolha de μ=0.01 é conservadora e adequada ao grau de heterogeneidade moderado a alto do dataset. O valor é repassado aos clientes via `fit_config` a cada round, garantindo que o termo proximal seja aplicado corretamente pelo `ai_client` mesmo sem conhecer a configuração global do servidor.

**Implementação no código:**

```python
# fl_server/server.py — build_base_strategy()
return FedProx(proximal_mu=proximal_mu, **common_kwargs)
```

### 1.3 Por que DP-SGD client-side (Opacus)

A Privacidade Diferencial é aplicada **no cliente**, antes de qualquer gradiente ou delta de peso sair do nó de borda. O mecanismo utiliza o Opacus (Yousefpour et al., 2021), que implementa DP-SGD com contabilização RDP:

**1. Clipping por amostra:** durante a passagem backward, Opacus intercepta o gradiente de cada amostra individualmente e o projeta para norma máxima `C₀ = 1.0`. O valor `C₀` é fixado antes de qualquer contato com os dados, baseado nos resultados empíricos de Yu et al. (2022) e Anil et al. (2022) para modelos de linguagem com LoRA. Isso garante a **independência de dados** necessária para a garantia formal `(ε, δ)`-DP.

**2. Ruído Gaussiano:** após o clipping por amostra e a agregação dos gradientes no batch, é adicionado ruído calibrado `N(0, σ²C₀²I)` com `σ = noise_multiplier`. O ruído é injetado nos gradientes das camadas LoRA antes do passo de otimização.

**3. Exclusão do modelo base:** os pesos do modelo base (Llama-3 em NF4 4-bit ou PubMedBERT) têm `requires_grad=False` e são completamente excluídos do mecanismo DP — somente as camadas LoRA são protegidas, o que mantém o custo computacional proporcional ao tamanho do adaptador, não do modelo completo.

**4. Contabilização RDP:** o accountant RDP (Mironov, 2017) rastreia o orçamento de privacidade acumulado. O subsampling de Poisson com taxa `q = 0.1` por round permite amplificação de privacidade. O `epsilon_cumulative` reportado no JSON de resultados é o valor a citar no artigo.

```python
# ai_client/fl_client.py — aplicação do Opacus DP-SGD
privacy_engine = PrivacyEngine()
model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
    module=lora_model,
    optimizer=optimizer,
    data_loader=train_loader,
    epochs=n_epochs,
    target_epsilon=target_epsilon,
    target_delta=FL_TARGET_DELTA,
    max_grad_norm=C0,   # C₀ = 1.0 (literature-based, data-independent)
)
```

**Por que client-side e não server-side?**

A DP server-side (adicionada *após* a agregação) não protege os gradientes individuais dos clientes enquanto estão em trânsito ou visíveis ao servidor — um servidor comprometido pode inspecionar os deltas antes de aplicar o ruído. Com DP client-side, o ruído já está incorporado no delta que sai do silo: mesmo que o servidor seja adversarial, ele recebe apenas gradientes privatizados. Essa é a garantia de privacidade mais forte no modelo de ameaça federado (Geyer et al., 2017; Wei et al., 2020).

O servidor (`fl_server`) é um agregador FedProx limpo — não adiciona ruído e não precisa ser confiável para que a garantia DP valha.

### 1.4 Modelo de ameaça mitigado

**Ataques de inversão de gradiente (Gradient/Model Inversion):** dado acesso aos deltas LoRA enviados por um cliente, um adversário pode tentar reconstruir o texto clínico que os originou (Zhu et al., 2019). O ruído Gaussiano injetado client-side corrompe os gradientes antes de deixarem o silo, tornando a reconstrução inviável para σ ≥ 1.0 — conforme demonstrado nos experimentos de inversão em `evaluation/src/evaluation/run_gradient_inversion.py`.

**Inferência de pertinência (Membership Inference):** um adversário tenta determinar se o registro de um paciente específico foi usado no treinamento. A garantia `(ε, δ)`-DP limita formalmente a vantagem do adversário. Com σ=0.9, 5 rounds e q=0.1, o ε cumulativo é da ordem de 4–5 (δ=1e-5), oferecendo proteção moderada a forte — adequada para dados clínicos anonimizados.

**Servidor adversarial:** como o ruído é aplicado client-side, um servidor comprometido que inspecione os deltas recebidos apenas vê gradientes já privatizados — a garantia DP não depende da integridade do servidor.

### 1.5 Limitações conhecidas

- **Secure Aggregation:** os deltas LoRA de cada cliente chegam ao servidor em texto claro (apenas privatizados por ruído). Para impedir que o servidor reconstrua contribuições individuais por diferença entre rounds, seria necessário Secure Aggregation criptográfica (Bonawitz et al., 2017). Essa extensão aumenta significativamente a complexidade operacional e está fora do escopo atual.
- **Neighboring relation — admissão-nível:** a garantia DP cobre a adição/remoção de uma admissão hospitalar completa (todos os DocumentReference, Condition e Patient associados). Pacientes com múltiplas admissões contribuem independentemente — o que é conservador e formalmente correto, mas pode subestimar a exposição de pacientes com histórico clínico extenso.
- **C₀ literatura vs. dados:** `C₀ = 1.0` é declarado antes de ver qualquer dado (garantia formal). O script de calibração (`FL_CALIBRATE_GRAD_NORM=true`) mede normas reais como verificação de sanidade; seu resultado não altera `C₀` em execuções formalmente DP.

---

## 2. Llama-3 quantizado em 4-bits + LoRA

### 2.1 A restrição de hardware como requisito de projeto

O sistema é projetado para rodar **localmente em uma única GPU de 12 GB de VRAM**, sem depender de infraestrutura de nuvem. Essa restrição não é arbitrária: ela reflete o hardware disponível em instituições de saúde de médio porte e é o que torna o sistema implantável em cenários reais.

O Llama-3-8B em precisão completa (float32) ocupa ~32 GB de VRAM — inviável. A combinação de quantização 4-bit + LoRA reduz esse requisito para **~6–8 GB durante o treinamento**, dentro da margem da GPU de 12 GB.

Estimativa do uso de VRAM:

| Componente | Precisão | VRAM aproximada |
|---|---|---|
| Llama-3-8B (pesos base, NF4 4-bit) | 4-bit NF4 | ~4.5 GB |
| Ativações + KV cache (seq=512) | bfloat16 | ~1.5 GB |
| Pesos LoRA treináveis (~24M params) | bfloat16 | ~0.2 GB |
| Gradientes LoRA + estados AdamW | float32 | ~1.0 GB |
| **Total estimado** | | **~7.2 GB** |

### 2.2 Quantização NF4 via BitsAndBytes

A quantização **NF4 (NormalFloat 4-bit)** representa cada parâmetro do modelo base em 4 bits usando um código de ponto flutuante normalizado otimizado para distribuições de pesos de redes neurais (aproximadamente gaussianas). A computação em tempo de inferência e treino ocorre em `bfloat16` — os pesos são desquantizados on-the-fly por bloco antes de cada operação de matriz.

**Dupla quantização** (`bnb_4bit_use_double_quant=True`): quantiza também as constantes de quantização dos blocos (que normalmente ficam em float32), economizando adicionalmente ~0.4 bits por parâmetro.

```python
# ai_client/model_setup.py — QuantizationConfig
BitsAndBytesConfig(
    load_in_4bit              = True,
    bnb_4bit_quant_type       = "nf4",       # NormalFloat 4-bit
    bnb_4bit_compute_dtype    = torch.bfloat16,
    bnb_4bit_use_double_quant = True,
)
```

O modelo base é carregado com `device_map="auto"`, deixando o BitsAndBytes alocar camadas na GPU disponível. Os pesos base são **completamente congelados** (`requires_grad=False`) — apenas os adaptadores LoRA são treináveis.

### 2.3 Por que LoRA e não fine-tuning completo

**Fine-tuning completo** de um LLM de 8B parâmetros exigiria:
- ~32 GB de VRAM para os pesos em float32
- ~64–96 GB para gradientes + estados do otimizador (Adam: 2 momentos × 32-bit)
- Transmissão de ~30 GB de dados por round por cliente no contexto federado

**LoRA** (Hu et al., 2022) decompõe a atualização de cada matriz de pesos em um produto de duas matrizes de posto baixo:

```
ΔW = A · B    onde A ∈ ℝ^(d×r), B ∈ ℝ^(r×k), r ≪ min(d, k)
```

Com `r = 16` (rank adotado), o número de parâmetros treináveis cai de **8B para ~24M** (~0.3% do total). Apenas as matrizes A e B são treinadas; o restante do modelo permanece em NF4 congelado.

A escala efetiva da adaptação é controlada por `lora_alpha`:

```
W_eff = W_base + (lora_alpha / r) · A · B = W_base + 2.0 · A · B
```

Com `alpha = 32` e `r = 16`, o fator de escala é 2.0 — valor padrão que balanceia estabilidade de treinamento e capacidade expressiva da adaptação.

### 2.4 Configuração dos adaptadores LoRA

```python
# ai_client/model_setup.py
LoraConfig(
    r            = 16,       # rank da decomposição
    lora_alpha   = 32,       # escala efetiva = 32/16 = 2.0
    lora_dropout = 0.05,     # regularização durante treino
    task_type    = TaskType.CAUSAL_LM,
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",   # atenção
        "gate_proj", "up_proj", "down_proj",        # FFN (SwiGLU)
    ],
    modules_to_save = ["embed_tokens", "lm_head"], # adaptados completamente
)
```

**Por que todos os 7 módulos?** Cobrir apenas `q_proj` e `v_proj` (configuração mínima comum) é suficiente para adaptação de estilo, mas insuficiente para uma nova tarefa de extração estruturada como CID-10. Incluir as projeções FFN (`gate_proj`, `up_proj`, `down_proj`) aumenta a capacidade de armazenar conhecimento factual de domínio médico nas camadas intermediárias.

**`modules_to_save`:** `embed_tokens` e `lm_head` são treinados completamente (não via LoRA) para que o modelo possa aprender o vocabulário específico de códigos CID-10 na camada de saída.

### 2.5 Eficiência de comunicação no contexto federado

O principal gargalo de escalabilidade em FL não é o cálculo — é a **comunicação**. Cada round envolve upload e download de pesos entre cliente e servidor. A escolha de LoRA transforma radicalmente esse custo:

| Modo de treinamento | Parâmetros transmitidos | Tamanho aproximado/round |
|---|---|---|
| Fine-tuning completo (bfloat16) | 8B | ~16 GB |
| LoRA rank=16 (bfloat16) | ~24M | ~48 MB |
| **Redução** | | **~333×** |

Na prática, apenas os tensores LoRA são serializados como `NDArrays` e enviados ao `fl_server` via gRPC. O modelo base Llama-3-8B nunca é transmitido — ele é carregado independentemente em cada nó a partir do cache HuggingFace (`model_cache` volume Docker).

Esse design é fundamental para viabilizar o protocolo federado em redes hospitalares com largura de banda limitada.

---

## 3. Referências

- McMahan, H. B. et al. (2017). *Communication-Efficient Learning of Deep Networks from Decentralized Data.* AISTATS 2017.
- Li, T. et al. (2020a). *Federated Learning on Non-IID Data Silos: An Experimental Study.* arXiv:2102.02079.
- Li, T. et al. (2020b). *Federated Optimization in Heterogeneous Networks.* MLSys 2020.
- Dwork, C. & Roth, A. (2014). *The Algorithmic Foundations of Differential Privacy.* Foundations and Trends in Theoretical Computer Science.
- Mironov, I. (2017). *Rényi Differential Privacy of the Gaussian Mechanism.* CSF 2017.
- Geyer, R. C. et al. (2017). *Differentially Private Federated Learning: A Client Level Perspective.* NeurIPS Workshop.
- Wei, K. et al. (2020). *Federated Learning with Differential Privacy: Algorithms and Performance Analysis.* IEEE TIFS.
- Yousefpour, A. et al. (2021). *Opacus: User-Friendly Differential Privacy Library in PyTorch.* arXiv:2109.12298.
- Yu, D. et al. (2022). *Differentially Private Fine-Tuning of Language Models.* ICLR 2022.
- Anil, R. et al. (2022). *Large-Scale Differentially Private BERT.* EMNLP 2022.
- Bonawitz, K. et al. (2017). *Practical Secure Aggregation for Privacy-Preserving Machine Learning.* CCS 2017.
- Zhu, L. et al. (2019). *Deep Leakage from Gradients.* NeurIPS 2019.
- Hu, E. J. et al. (2022). *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR 2022.
- Dettmers, T. et al. (2023). *QLoRA: Efficient Finetuning of Quantized LLMs.* NeurIPS 2023.
- HL7 International. *FHIR R4 Specification.* https://hl7.org/fhir/R4.
