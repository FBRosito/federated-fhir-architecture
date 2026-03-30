"""
server.py
---------
Orquestrador Flower para o pipeline de Aprendizado Federado sobre dados FHIR.

Estratégia: FedProx (mais robusto que FedAvg para dados Non-IID) envolvida por
DifferentialPrivacyServerSideAdaptiveClipping — o servidor aprende adaptativamente
a norma de clipping ideal e adiciona ruído Gaussiano calibrado aos pesos
agregados antes de redistribuí-los, mitigando ataques de inversão de gradiente.

Por que FedProx para Non-IID?
    Nossos dados são deliberadamente heterogêneos (partições por especialidade
    médica). FedProx introduz um termo proximal μ||w - w_global||² que penaliza
    clientes que se desviam muito do modelo global, estabilizando a convergência.

Por que DP adaptativa?
    A norma de clipping ótima depende da magnitude real das atualizações —
    desconhecida a priori. A variante adaptativa ajusta a norma por round com
    base num quantil-alvo da distribuição de normas observadas, dispensando
    busca manual de hiperparâmetro.

Modelo de ameaça mitigado:
    - Inversão de gradiente (Gradient Inversion / Model Inversion Attacks):
      o ruído Gaussiano adicionado no servidor impede a reconstrução exata
      dos dados de treinamento a partir dos pesos recebidos.
    - Membership Inference: a garantia (ε, δ)-DP limita a capacidade de um
      adversário distinguir se um indivíduo participou do treinamento.

Limitação conhecida:
    A DP-server-side não protege os pesos *em trânsito* do cliente para o
    servidor — os pesos individuais por cliente chegam sem ruído. Para
    proteção end-to-end, combine com DP-client-side (local DP) ou Secure
    Aggregation.

Variáveis de ambiente:
    FL_SERVER_ADDRESS      Endereço de escuta gRPC (default: [::]:9091)
    FL_NUM_ROUNDS          Rounds de treinamento federado (default: 5)
    FL_MIN_CLIENTS         Clientes mínimos para iniciar cada round (default: 2)
    FL_STRATEGY            "fedprox" | "fedavg" (default: fedprox)
    FL_PROXIMAL_MU         Coeficiente proximal μ do FedProx (default: 0.01)
    FL_NOISE_MULTIPLIER    Multiplicador de ruído DP σ/C (default: 0.9)
    FL_INITIAL_CLIP_NORM   Norma de clipping inicial (default: 1.0)
    FL_CLIP_NORM_TARGET_Q  Quantil-alvo de clipping adaptativo (default: 0.5)
    FL_FRACTION_FIT        Fração de clientes usados por round de treino (default: 1.0)
    FL_FRACTION_EVAL       Fração de clientes usados por round de avaliação (default: 1.0)
"""

from __future__ import annotations

import logging
import math
import os

import flwr as fl
from flwr.common import NDArrays, Scalar, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig, start_server
from flwr.server.strategy import FedAvg, FedProx, Strategy
from flwr.server.strategy import (
    DifferentialPrivacyServerSideAdaptiveClipping as DPAdaptiveClipping,
)
from flwr.server.strategy import (
    DifferentialPrivacyServerSideFixedClipping as DPFixedClipping,
)

log = logging.getLogger(__name__)

# ── Leitura de configuração via ambiente ──────────────────────────────────────

def _env_float(key: str, default: float) -> float:
    return float(os.getenv(key, default))

def _env_int(key: str, default: int) -> int:
    return int(os.getenv(key, default))

def _env_str(key: str, default: str) -> str:
    return os.getenv(key, default)


SERVER_ADDRESS      = _env_str("FL_SERVER_ADDRESS", "[::]:9091")
NUM_ROUNDS          = _env_int("FL_NUM_ROUNDS", 5)
MIN_CLIENTS         = _env_int("FL_MIN_CLIENTS", 2)
STRATEGY_NAME       = _env_str("FL_STRATEGY", "fedprox").lower()
PROXIMAL_MU         = _env_float("FL_PROXIMAL_MU", 0.01)
NOISE_MULTIPLIER    = _env_float("FL_NOISE_MULTIPLIER", 0.9)
INITIAL_CLIP_NORM   = _env_float("FL_INITIAL_CLIP_NORM", 1.0)
CLIP_NORM_TARGET_Q  = _env_float("FL_CLIP_NORM_TARGET_Q", 0.5)
FRACTION_FIT        = _env_float("FL_FRACTION_FIT", 1.0)
FRACTION_EVAL       = _env_float("FL_FRACTION_EVAL", 1.0)

# ── Estimativa do orçamento de privacidade (Gaussian Mechanism approx.) ───────

def estimate_privacy_budget(
    noise_multiplier: float,
    num_rounds: int,
    num_clients_per_round: int,
    total_clients: int,
    delta: float = 1e-5,
) -> float:
    """
    Estimativa do gasto de ε por composição sequencial simples do mecanismo
    Gaussiano ao longo dos rounds.

    AVISO: Esta é uma cota superior conservadora. Para contabilidade exata
    use a biblioteca `autodp` (Rényi DP) ou `opacus` (PRV accountant).

    Args:
        noise_multiplier:        σ / C (razão ruído/sensibilidade).
        num_rounds:              Número de rounds federados.
        num_clients_per_round:   Clientes amostrados por round.
        total_clients:           Total de clientes no pool.
        delta:                   Probabilidade de falha da garantia DP.

    Returns:
        Estimativa do ε total consumido.
    """
    if noise_multiplier <= 0:
        return float("inf")

    sampling_rate = num_clients_per_round / max(total_clients, 1)
    # Cota por round via mecanismo Gaussiano analítico aproximado
    epsilon_per_round = math.sqrt(2 * math.log(1.25 / delta)) / noise_multiplier
    # Composição sequencial simples (conservadora): ε_total ≈ T * ε_round * q
    epsilon_total = num_rounds * epsilon_per_round * sampling_rate
    return round(epsilon_total, 4)


# ── Construtores de estratégia ────────────────────────────────────────────────

def build_base_strategy(
    strategy_name: str,
    min_clients: int,
    fraction_fit: float,
    fraction_eval: float,
    proximal_mu: float,
    fit_metrics_fn,
    eval_metrics_fn,
) -> Strategy:
    """Instancia FedProx ou FedAvg conforme `strategy_name`."""
    common_kwargs = dict(
        fraction_fit                  = fraction_fit,
        fraction_evaluate             = fraction_eval,
        min_fit_clients               = min_clients,
        min_evaluate_clients          = max(1, min_clients // 2),
        min_available_clients         = min_clients,
        on_fit_config_fn              = make_fit_config_fn(proximal_mu),
        on_evaluate_config_fn         = make_eval_config_fn(),
        fit_metrics_aggregation_fn    = fit_metrics_fn,
        evaluate_metrics_aggregation_fn = eval_metrics_fn,
        accept_failures               = True,
        # initial_parameters=None: Flower solicitará parâmetros iniciais
        # ao primeiro cliente disponível no round 0
        initial_parameters            = None,
    )

    if strategy_name == "fedprox":
        log.info("Estratégia base: FedProx (μ=%.4f)", proximal_mu)
        return FedProx(proximal_mu=proximal_mu, **common_kwargs)

    log.info("Estratégia base: FedAvg")
    return FedAvg(**common_kwargs)


def wrap_with_dp(
    base_strategy: Strategy,
    num_sampled_clients: int,
    noise_multiplier: float,
    initial_clip_norm: float,
    target_quantile: float,
) -> DPAdaptiveClipping:
    """
    Envolve a estratégia base com DP server-side de clipping adaptativo.

    Mecanismo:
        1. Ao receber os parâmetros de cada cliente, o servidor os clippa
           na norma L2 C_t (atualizada adaptativamente a cada round).
        2. Após a agregação (FedAvg/FedProx), adiciona ruído Gaussiano
           N(0, σ²I) com σ = noise_multiplier × C_t ao vetor agregado.
        3. C_t é ajustado para que a fração `target_quantile` das normas
           dos clientes fique abaixo do limiar — sem exigir conhecimento
           prévio da distribuição das atualizações.

    Args:
        base_strategy:       Estratégia base já configurada.
        num_sampled_clients: Clientes amostrados por round (afeta σ_count).
        noise_multiplier:    σ / C — razão ruído/sensibilidade.
        initial_clip_norm:   Norma de clipping inicial C_0.
        target_quantile:     Quantil-alvo de clipping (0.5 = mediana).

    Returns:
        Estratégia com DP adaptativa configurada.
    """
    # clipped_count_stddev: desvio padrão do ruído na contagem de clientes
    # clipados — deve ser grande o suficiente para que o noise_multiplier
    # efetivo do mecanismo de contagem seja alcançável.
    # Regra prática segura: usar sqrt(num_sampled_clients)
    clipped_count_stddev = max(1.0, math.sqrt(num_sampled_clients))

    dp_strategy = DPAdaptiveClipping(
        strategy              = base_strategy,
        noise_multiplier      = noise_multiplier,
        num_sampled_clients   = num_sampled_clients,
        initial_clipping_norm = initial_clip_norm,
        target_clipped_quantile = target_quantile,
        clip_norm_lr          = 0.2,      # learning rate do ajuste adaptativo de C_t
        clipped_count_stddev  = clipped_count_stddev,
    )
    log.info(
        "DP adaptativa configurada: σ=%.2f, C_0=%.2f, q_target=%.2f, "
        "σ_count=%.2f, clientes/round=%d",
        noise_multiplier, initial_clip_norm, target_quantile,
        clipped_count_stddev, num_sampled_clients,
    )
    return dp_strategy


# ── Callbacks de configuração por round ──────────────────────────────────────

def make_fit_config_fn(proximal_mu: float):
    """
    Retorna função que gera a configuração de treinamento enviada a cada cliente
    no início de cada round de `fit`.

    O campo `proximal_mu` é repassado ao cliente para que ele aplique o termo
    proximal correto mesmo sem conhecer a configuração global do servidor.
    """
    def fit_config(server_round: int) -> dict[str, Scalar]:
        config: dict[str, Scalar] = {
            "server_round":  server_round,
            "proximal_mu":   proximal_mu,
            # Decaimento do learning rate a partir do round 3
            "learning_rate": 2e-4 if server_round <= 2 else 1e-4,
            "num_epochs":    1,
        }
        log.debug("fit_config round %d: %s", server_round, config)
        return config
    return fit_config


def make_eval_config_fn():
    """Retorna função que gera a configuração de avaliação por round."""
    def eval_config(server_round: int) -> dict[str, Scalar]:
        return {
            "server_round": server_round,
            "compute_accuracy": True,
        }
    return eval_config


# ── Agregação de métricas ─────────────────────────────────────────────────────

def aggregate_fit_metrics(
    metrics: list[tuple[int, dict[str, Scalar]]],
) -> dict[str, Scalar]:
    """
    Agrega métricas de treinamento reportadas pelos clientes (média ponderada
    pelo número de exemplos usados em cada cliente).
    """
    if not metrics:
        return {}
    total_examples = sum(n for n, _ in metrics)
    if total_examples == 0:
        return {}

    aggregated: dict[str, float] = {}
    for n, m in metrics:
        weight = n / total_examples
        for key, value in m.items():
            aggregated[key] = aggregated.get(key, 0.0) + float(value) * weight

    log.info(
        "Métricas fit agregadas (%d clientes): %s",
        len(metrics),
        {k: f"{v:.4f}" for k, v in aggregated.items()},
    )
    return {k: round(v, 6) for k, v in aggregated.items()}


def aggregate_eval_metrics(
    metrics: list[tuple[int, dict[str, Scalar]]],
) -> dict[str, Scalar]:
    """Agrega métricas de avaliação (média ponderada por exemplos)."""
    if not metrics:
        return {}
    total_examples = sum(n for n, _ in metrics)
    if total_examples == 0:
        return {}

    aggregated: dict[str, float] = {}
    for n, m in metrics:
        weight = n / total_examples
        for key, value in m.items():
            aggregated[key] = aggregated.get(key, 0.0) + float(value) * weight

    log.info(
        "Métricas eval agregadas (%d clientes): %s",
        len(metrics),
        {k: f"{v:.4f}" for k, v in aggregated.items()},
    )
    return {k: round(v, 6) for k, v in aggregated.items()}


# ── Construção da estratégia completa ─────────────────────────────────────────

def build_strategy(
    strategy_name: str  = STRATEGY_NAME,
    min_clients: int    = MIN_CLIENTS,
    num_rounds: int     = NUM_ROUNDS,
    fraction_fit: float = FRACTION_FIT,
    fraction_eval: float = FRACTION_EVAL,
    proximal_mu: float  = PROXIMAL_MU,
    noise_multiplier: float = NOISE_MULTIPLIER,
    initial_clip_norm: float = INITIAL_CLIP_NORM,
    target_quantile: float  = CLIP_NORM_TARGET_Q,
) -> Strategy:
    """
    Constrói a estratégia final: base (FedProx|FedAvg) + DP adaptativa.

    Também loga a estimativa do orçamento de privacidade consumido ao longo
    de todos os rounds.
    """
    num_sampled = max(1, round(min_clients * fraction_fit))

    base = build_base_strategy(
        strategy_name  = strategy_name,
        min_clients    = min_clients,
        fraction_fit   = fraction_fit,
        fraction_eval  = fraction_eval,
        proximal_mu    = proximal_mu,
        fit_metrics_fn = aggregate_fit_metrics,
        eval_metrics_fn = aggregate_eval_metrics,
    )

    strategy = wrap_with_dp(
        base_strategy       = base,
        num_sampled_clients = num_sampled,
        noise_multiplier    = noise_multiplier,
        initial_clip_norm   = initial_clip_norm,
        target_quantile     = target_quantile,
    )

    # Estimativa do orçamento de privacidade
    eps = estimate_privacy_budget(
        noise_multiplier        = noise_multiplier,
        num_rounds              = num_rounds,
        num_clients_per_round   = num_sampled,
        total_clients           = min_clients,
        delta                   = 1e-5,
    )
    log.info(
        "Orçamento de privacidade estimado (δ=1e-5): ε ≈ %.4f "
        "ao longo de %d rounds com σ=%.2f [cota superior conservadora]",
        eps, num_rounds, noise_multiplier,
    )
    if eps > 10.0:
        log.warning(
            "ε=%.2f é alto — privacidade fraca. Considere aumentar "
            "noise_multiplier (FL_NOISE_MULTIPLIER) ou reduzir num_rounds.",
            eps,
        )

    return strategy


# ── ServerApp (API moderna do Flower 1.x) ─────────────────────────────────────

def server_fn(context) -> ServerAppComponents:
    """
    Função de fábrica do ServerApp. Chamada pelo runtime Flower ao iniciar.

    Lê os hiperparâmetros do `context.run_config` (definidos no ficheiro
    `pyproject.toml` da run, se presente) com fallback para as variáveis
    de ambiente.
    """
    run_cfg = context.run_config if hasattr(context, "run_config") else {}

    strategy = build_strategy(
        strategy_name    = str(run_cfg.get("strategy",        STRATEGY_NAME)),
        min_clients      = int(run_cfg.get("min_clients",     MIN_CLIENTS)),
        num_rounds       = int(run_cfg.get("num_rounds",      NUM_ROUNDS)),
        fraction_fit     = float(run_cfg.get("fraction_fit",  FRACTION_FIT)),
        fraction_eval    = float(run_cfg.get("fraction_eval", FRACTION_EVAL)),
        proximal_mu      = float(run_cfg.get("proximal_mu",   PROXIMAL_MU)),
        noise_multiplier = float(run_cfg.get("noise_multiplier", NOISE_MULTIPLIER)),
        initial_clip_norm = float(run_cfg.get("initial_clip_norm", INITIAL_CLIP_NORM)),
        target_quantile  = float(run_cfg.get("target_quantile", CLIP_NORM_TARGET_Q)),
    )

    config = ServerConfig(
        num_rounds    = int(run_cfg.get("num_rounds", NUM_ROUNDS)),
        round_timeout = float(run_cfg.get("round_timeout", 300.0)),
    )

    return ServerAppComponents(strategy=strategy, config=config)


# ServerApp — ponto de entrada para `flwr run` / SuperLink
app = ServerApp(server_fn=server_fn)


# ── Ponto de entrada legado (start_server) ────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level   = logging.INFO,
        format  = "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt = "%Y-%m-%dT%H:%M:%S",
    )

    log.info("=== FL Server iniciando ===")
    log.info(
        "Configuração: strategy=%s | rounds=%d | min_clients=%d | "
        "σ=%.2f | C_0=%.2f | μ=%.4f",
        STRATEGY_NAME, NUM_ROUNDS, MIN_CLIENTS,
        NOISE_MULTIPLIER, INITIAL_CLIP_NORM, PROXIMAL_MU,
    )

    strategy = build_strategy()

    history = start_server(
        server_address         = SERVER_ADDRESS,
        config                 = ServerConfig(num_rounds=NUM_ROUNDS, round_timeout=300.0),
        strategy               = strategy,
        grpc_max_message_length = 512 * 1024 * 1024,  # 512 MB — LLM weights são grandes
    )

    # Resumo pós-treinamento
    log.info("=== Treinamento federado concluído ===")
    if history.metrics_distributed_fit:
        rounds = sorted(history.metrics_distributed_fit.keys())
        for metric_name in rounds:
            entries = history.metrics_distributed_fit[metric_name]
            log.info("  fit/%s: %s", metric_name, entries)
    if history.losses_distributed:
        log.info("  losses_distributed: %s", history.losses_distributed)


if __name__ == "__main__":
    main()
