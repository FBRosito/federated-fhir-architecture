# Decisões de Arquitetura — FL-FHIR Architecture

**Projeto:** Aprendizado Federado sobre Dados Clínicos no Padrão FHIR
**Repositório:** `fl_fhir_architecture`
**Versão do documento:** 2.0
**Data:** 2026-03-29

---

## Sumário

1. [FedProx + Privacidade Diferencial Adaptativa](#1-fedprox--privacidade-diferencial-adaptativa)
   - 1.1 [O problema: dados Non-IID em ambientes hospitalares](#11-o-problema-dados-non-iid-em-ambientes-hospitalares)
   - 1.2 [Por que FedProx e não FedAvg](#12-por-que-fedprox-e-não-fedavg)
   - 1.3 [Por que Privacidade Diferencial Adaptativa](#13-por-que-privacidade-diferencial-adaptativa)
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

## 1. FedProx + Privacidade Diferencial Adaptativa

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

### 1.3 Por que Privacidade Diferencial Adaptativa

A DP (Privacidade Diferencial) server-side envolve a estratégia base em dois mecanismos sequenciais aplicados às atualizações LoRA recebidas de cada cliente:

**1. Clipping L2:** cada vetor de pesos é truncado para ter norma máxima `C_t`, limitando a influência de qualquer cliente individual na agregação (sensibilidade global).

**2. Ruído Gaussiano:** após a agregação, é adicionado ruído `N(0, σ²I)` com `σ = noise_multiplier × C_t` ao vetor agregado antes de redistribuí-lo.

A combinação garante a propriedade `(ε, δ)`-DP: para `δ = 1e-5`, o orçamento de privacidade consumido ao longo de `T` rounds é estimado por:

```
ε_total ≈ T · √(2 · ln(1.25/δ)) / σ · (n_round / n_total)
```

**Por que adaptativa e não clipping fixo?**

A variante com clipping fixo (`DifferentialPrivacyServerSideFixedClipping`) exige que o operador defina `C` manualmente. Escolher `C` muito grande reduz a efetividade da DP; muito pequeno destrói informação útil dos gradientes. A norma ideal depende da magnitude real das atualizações — desconhecida a priori e variável entre rounds.

A variante **adaptativa** (`DifferentialPrivacyServerSideAdaptiveClipping`) resolve isso aprendendo `C_t` por round. O servidor mantém a estimativa do quantil-alvo `q = 0.5` (mediana) da distribuição de normas dos clientes e ajusta `C_t` com uma taxa de aprendizado `lr = 0.2`:

```python
# fl_server/server.py — wrap_with_dp()
dp_strategy = DPAdaptiveClipping(
    strategy                 = base_strategy,
    noise_multiplier         = 0.9,       # σ/C — razão ruído/sensibilidade
    num_sampled_clients      = n,
    initial_clipping_norm    = 1.0,       # C_0
    target_clipped_quantile  = 0.5,       # mediana das normas
    clip_norm_lr             = 0.2,       # taxa de ajuste de C_t
    clipped_count_stddev     = √n,        # ruído na contagem de clientes clipados
)
```

O resultado é que o sistema **dispensa busca manual de hiperparâmetro de clipping** e se adapta automaticamente à magnitude real das atualizações LoRA — que pode variar significativamente nas primeiras rounds enquanto o modelo ainda está instável.

### 1.4 Modelo de ameaça mitigado

**Ataques de inversão de gradiente (Gradient/Model Inversion):** dado acesso aos pesos enviados por um cliente, um adversário pode tentar reconstruir os dados de treinamento que os geraram. O ruído Gaussiano adicionado pelo servidor corrompido intencionalmente os pesos antes da redistribuição, impedindo a reconstrução exata mesmo que o adversário intercepte o modelo global.

**Inferência de pertinência (Membership Inference):** um adversário tenta determinar se um registro específico de paciente foi utilizado no treinamento. A garantia `(ε, δ)`-DP limita formalmente a vantagem do adversário: para `ε ≈ 1.5` (estimativa conservadora com σ=0.9, T=5, q=0.5), o adversário tem ganho de informação estritamente limitado.

### 1.5 Limitações conhecidas

A DP implementada é **server-side**: o ruído é adicionado *após* a agregação, sobre os pesos agregados. Os pesos individuais enviados por cada cliente chegam ao servidor sem proteção de ruído local. Para proteção *end-to-end*, seria necessário combinar com:

- **DP client-side (DP local):** cada cliente adiciona ruído antes de enviar suas atualizações.
- **Secure Aggregation:** os pesos são agregados de forma criptograficamente segura sem que o servidor veja os valores individuais.

Essas extensões não foram implementadas por aumentarem significativamente a complexidade operacional e o custo computacional — e estão fora do escopo da prova de conceito atual.

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
- Andrew, G. et al. (2021). *Differentially Private Learning with Adaptive Clipping.* NeurIPS 2021.
- Hu, E. J. et al. (2022). *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR 2022.
- Dettmers, T. et al. (2023). *QLoRA: Efficient Finetuning of Quantized LLMs.* NeurIPS 2023.
- HL7 International. *FHIR R5 Specification.* https://hl7.org/fhir/R5.
