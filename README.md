# FL-FHIR Architecture
## Aprendizado Federado sobre Dados Clínicos no Padrão HL7 FHIR

---

## Visão Geral e Mérito Científico

Este projeto investiga a viabilidade de treinar **Modelos de Linguagem de Grande Escala (LLMs)** de forma federada sobre dados clínicos estruturados no padrão **HL7 FHIR R5**, preservando a privacidade dos pacientes sem centralizar os registros médicos.

A intersecção das três áreas constitui a contribuição principal:

| Domínio | Papel no sistema |
|---|---|
| **LLMs (Llama-3-8B + LoRA)** | Extração de códigos CID-10 a partir de evoluções clínicas em linguagem natural |
| **Aprendizado Federado (Flower / FedProx)** | Treinamento distribuído sem que os dados saiam dos nós clientes; apenas deltas de pesos LoRA (~3–6 MB/round) trafegam na rede |
| **HL7 FHIR R5** | Camada de interoperabilidade semântica que padroniza os recursos clínicos (Condition, DocumentReference, Patient) e viabiliza a portabilidade entre instituições |

**Problema científico endereçado:** dados hospitalares reais são inerentemente heterogêneos entre unidades de saúde (distribuição Non-IID). O sistema emprega **FedProx** com **Privacidade Diferencial Adaptativa server-side** para lidar com essa heterogeneidade sem comprometer as garantias formais de privacidade (ε, δ)-DP.

---

## Arquitetura de Microserviços

```
┌──────────────────────────────────────────────────────────────────┐
│                          fl_server                               │
│  FedProx (μ=0.01)  ←  DP Adaptativa (σ=0.9, C₀=1.0, q=0.5)    │
│  gRPC :9091                                                      │
└────────────┬────────────────────────────────┬────────────────────┘
             │  pesos LoRA globais (NDArrays)  │
             ▼                                 ▼
┌────────────────────┐               ┌────────────────────┐
│  ai_client  (A)    │               │  ai_client  (B)    │
│  Partição 0        │               │  Partição 1        │
│  Cardiologia       │               │  Pneumologia       │
│  Llama-3-8B NF4    │               │  Llama-3-8B NF4    │
│  LoRA rank=16      │               │  LoRA rank=16      │
└────────┬───────────┘               └───────────┬────────┘
         │  HTTP GET /fhir                        │
         └──────────────────┬────────────────────┘
                            ▼
              ┌─────────────────────────┐
              │       hapi_fhir         │
              │  FHIR R5 — porta 8080   │
              │  Condition              │
              │  DocumentReference      │
              │  Patient                │
              └──────────┬──────────────┘
                         ▲
              ┌──────────┴──────────────┐
              │      etl_worker         │
              │  CSV → FHIR Bundle      │
              │  Particionamento Non-IID│
              └─────────────────────────┘
```

### Componentes

**`hapi_fhir`** — Servidor FHIR R5 (HAPI FHIR). Armazena e valida semanticamente os recursos clínicos. É o único ponto de troca de dados entre o pipeline de ingestão e os clientes de IA, garantindo a separação de contextos.

**`etl_worker`** — Pipeline de transformação CSV → FHIR. Mapeia diagnósticos para CID-10 e cria Transaction Bundles com quatro recursos por paciente (Patient, Condition, Composition, DocumentReference). Implementa o particionamento Non-IID por especialidade médica (cardiologia, pneumologia, endocrinologia, geral).

**`ai_client`** — Cliente Flower (`NumPyClient`). Consome recursos FHIR estruturados via API REST, treina o modelo localmente por um epoch por round e retorna apenas os pesos LoRA atualizados. O modelo base (Llama-3-8B) nunca sai do nó. Utiliza lazy loading para evitar OOM em GPUs de 12 GB VRAM.

**`fl_server`** — Orquestrador Flower (`ServerApp`). Executa a estratégia FedProx envelopada por Privacidade Diferencial Adaptativa. Agrega os pesos LoRA dos clientes, adiciona ruído Gaussiano calibrado e redistribui o modelo global a cada round.

---

## Pré-requisitos

| Dependência | Versão mínima | Finalidade |
|---|---|---|
| [Docker](https://docs.docker.com/get-docker/) + [Docker Compose](https://docs.docker.com/compose/) | 24.0 / 2.20 | Orquestração dos serviços |
| [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) | 1.14 | Passagem da GPU para o container `ai_client` |
| [uv](https://docs.astral.sh/uv/getting-started/installation/) | 0.5+ | Gerenciamento de pacotes e ambientes virtuais |
| GPU NVIDIA com CUDA ≥ 12.4 | **≥ 12 GB VRAM** | Inferência + fine-tuning do Llama-3-8B quantizado |
| Python | 3.13 | Runtime (gerenciado pelo uv) |
| Token HuggingFace | — | Acesso ao modelo `meta-llama/Meta-Llama-3-8B-Instruct` (gated) |

> **Atenção:** o serviço `ai_client` é o único que exige GPU. Os demais serviços (`fl_server`, `etl_worker`, `hapi_fhir`) rodam em CPU.

---

## Guia de Reprodução Passo a Passo

### 1. Clonar o repositório e instalar dependências

```bash
git clone <url-do-repositorio>
cd fl_fhir_architecture

# Instala todas as dependências do workspace a partir do lockfile
uv sync --frozen
```

### 2. Configurar o token HuggingFace

O modelo Llama-3-8B-Instruct é *gated* e requer autenticação:

```bash
# Crie um arquivo .env na raiz do projeto
cat > .env <<'EOF'
HF_TOKEN=hf_seu_token_aqui
EOF
```

### 3. Validar o pipeline ETL localmente (dry-run)

Antes de subir o stack completo, valide que os bundles FHIR são gerados corretamente:

```bash
uv run python etl_worker/etl_pipeline.py \
  --data etl_worker/data/clinical_evolutions.csv \
  --partition 0 \
  --fhir-url http://localhost:8080/fhir \
  --dry-run
```

### 4. Subir o stack completo com Docker Compose

```bash
docker-compose up --build
```

A ordem de inicialização é gerenciada pelos health checks:
1. `hapi_fhir` — aguarda o endpoint `/fhir/metadata` responder (30 s de intervalo)
2. `etl_worker` — inicia após `hapi_fhir` estar saudável; popula o servidor FHIR
3. `fl_server` — aguarda `hapi_fhir`; expõe gRPC na porta 9091
4. `ai_client` — aguarda `fl_server` e `hapi_fhir`; inicia o treinamento federado

### 5. Acompanhar o treinamento

```bash
# Logs do servidor FL (rounds, métricas agregadas, orçamento DP)
docker-compose logs -f fl_server

# Logs do cliente de IA (loss, perplexidade por round)
docker-compose logs -f ai_client

# Métricas em CSV (atualizado a cada round)
tail -f evaluation/logs/run.csv
```

### 6. Executar serviços individualmente (desenvolvimento)

```bash
# Servidor FL
uv run fl-server

# Cliente de IA
uv run ai-client

# ETL worker
uv run etl-worker

# Logger de métricas
uv run evaluation
```

### 7. Ajustar hiperparâmetros via variáveis de ambiente

```bash
# Exemplo: aumentar rounds e reduzir ruído DP
FL_NUM_ROUNDS=10 FL_NOISE_MULTIPLIER=0.5 FL_MIN_CLIENTS=4 uv run fl-server
```

Veja a tabela completa de variáveis em [CLAUDE.md](./CLAUDE.md).

---

## Estrutura do Repositório

```
fl_fhir_architecture/
├── fl_server/          # Orquestrador Flower (FedProx + DP)
├── ai_client/          # Cliente FL com Llama-3 + LoRA
├── etl_worker/         # Pipeline CSV → FHIR
├── evaluation/         # Logger de métricas federadas
├── docs/               # Decisões de arquitetura
├── docker-compose.yml
└── pyproject.toml      # Workspace uv (raiz)
```

---

## Referências

- McMahan, B. et al. (2017). *Communication-Efficient Learning of Deep Networks from Decentralized Data.* AISTATS.
- Li, T. et al. (2020). *Federated Optimization in Heterogeneous Networks (FedProx).* MLSys.
- Hu, E. et al. (2022). *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR.
- Dwork, C. & Roth, A. (2014). *The Algorithmic Foundations of Differential Privacy.* FnTCS.
- HL7 International. *FHIR R5 Specification.* hl7.org/fhir/R5.
- Flower (flwr). *A Friendly Federated Learning Framework.* flower.ai.
