"""
fhir_benchmark.py
-----------------
Benchmark de latência e completeness do servidor HAPI FHIR.

Métricas:
  - Latência POST (upload de bundles): P50, P95, P99
  - Latência GET (busca paginada): P50, P95, P99
  - Completeness score: fração de recursos esperados encontrados no servidor
  - Throughput: bundles/segundo para carga em lote

Uso:
    uv run python evaluation/fhir_benchmark.py --fhir-url http://localhost:8080/fhir --n-samples 100

Nota: executar LOCALMENTE para validar código e tuning; números publicáveis
devem vir de um ambiente cloud com infra dedicada para eliminar ruído de WSL.
"""

from __future__ import annotations

import argparse
import logging
import statistics
import time
import os
from dataclasses import dataclass, field
from pathlib import Path

import httpx

log = logging.getLogger(__name__)

_DEFAULT_FHIR_URL = os.getenv("FHIR_SERVER_URL", "http://localhost:8080/fhir")
_REQUEST_TIMEOUT  = 60.0


@dataclass
class FHIRBenchmarkResult:
    """Resultados do benchmark FHIR."""
    post_latencies_ms:  list[float] = field(default_factory=list)
    get_latencies_ms:   list[float] = field(default_factory=list)
    completeness_score: float = 0.0
    n_bundles_posted:   int = 0
    n_resources_found:  int = 0
    n_resources_expected: int = 0
    errors:             list[str] = field(default_factory=list)

    def _percentiles(self, latencies: list[float]) -> dict[str, float]:
        if not latencies:
            return {"p50": float("nan"), "p95": float("nan"), "p99": float("nan"), "mean": float("nan")}
        s = sorted(latencies)
        n = len(s)
        return {
            "p50":  s[int(n * 0.50)],
            "p95":  s[int(n * 0.95)],
            "p99":  s[min(int(n * 0.99), n - 1)],
            "mean": statistics.mean(latencies),
        }

    def post_stats(self) -> dict[str, float]:
        return self._percentiles(self.post_latencies_ms)

    def get_stats(self) -> dict[str, float]:
        return self._percentiles(self.get_latencies_ms)

    def __str__(self) -> str:
        p = self.post_stats()
        g = self.get_stats()
        return (
            f"FHIR Benchmark Results\n"
            f"  POST (n={len(self.post_latencies_ms)}): "
            f"mean={p['mean']:.1f}ms P50={p['p50']:.1f}ms P95={p['p95']:.1f}ms P99={p['p99']:.1f}ms\n"
            f"  GET  (n={len(self.get_latencies_ms)}): "
            f"mean={g['mean']:.1f}ms P50={g['p50']:.1f}ms P95={g['p95']:.1f}ms P99={g['p99']:.1f}ms\n"
            f"  Completeness: {self.completeness_score:.2%} "
            f"({self.n_resources_found}/{self.n_resources_expected} recursos)\n"
            f"  Errors: {len(self.errors)}"
        )


def benchmark_post_bundles(
    fhir_url: str,
    bundles_dir: Path,
    n_samples: int = 50,
) -> FHIRBenchmarkResult:
    """
    Mede latência de POST para Transaction Bundles existentes no bundles_dir.

    Apenas lê bundles já gerados pelo mimic_builder — não gera novos dados.
    Bundles são enviados um por vez (sem paralelismo) para isolar a latência.

    Args:
        fhir_url:    URL base do HAPI FHIR.
        bundles_dir: Diretório com bundle_NNNNNN.json.
        n_samples:   Número de bundles a enviar.

    Returns:
        FHIRBenchmarkResult com latências de POST.
    """
    result = FHIRBenchmarkResult()
    bundle_files = sorted(bundles_dir.glob("bundle_*.json"))[:n_samples]

    if not bundle_files:
        log.warning("Nenhum bundle encontrado em %s.", bundles_dir)
        return result

    log.info("Benchmark POST: %d bundles em %s...", len(bundle_files), fhir_url)

    with httpx.Client(timeout=_REQUEST_TIMEOUT) as client:
        for bf in bundle_files:
            payload = bf.read_bytes()
            t0 = time.perf_counter()
            try:
                resp = client.post(
                    f"{fhir_url.rstrip('/')}",
                    content=payload,
                    headers={"Content-Type": "application/fhir+json"},
                )
                elapsed_ms = (time.perf_counter() - t0) * 1000
                if resp.status_code < 400:
                    result.post_latencies_ms.append(elapsed_ms)
                    result.n_bundles_posted += 1
                else:
                    result.errors.append(f"POST {bf.name}: HTTP {resp.status_code}")
            except Exception as exc:
                result.errors.append(f"POST {bf.name}: {exc}")

    log.info("POST benchmark: %d bundles | stats: %s", result.n_bundles_posted, result.post_stats())
    return result


def benchmark_get_resources(
    fhir_url: str,
    resource_types: list[str] | None = None,
    n_pages: int = 10,
) -> FHIRBenchmarkResult:
    """
    Mede latência de GET para buscas paginadas de recursos FHIR.

    Args:
        fhir_url:        URL base do HAPI FHIR.
        resource_types:  Lista de tipos a buscar (default: Patient, Condition, DocumentReference).
        n_pages:         Número de páginas a buscar por tipo.

    Returns:
        FHIRBenchmarkResult com latências de GET.
    """
    if resource_types is None:
        resource_types = ["Patient", "Condition", "DocumentReference"]

    result = FHIRBenchmarkResult()
    base = fhir_url.rstrip("/")

    with httpx.Client(timeout=_REQUEST_TIMEOUT) as client:
        for rtype in resource_types:
            url: str | None = f"{base}/{rtype}?_count=50"
            page = 0
            while url and page < n_pages:
                t0 = time.perf_counter()
                try:
                    resp = client.get(url, timeout=_REQUEST_TIMEOUT)
                    elapsed_ms = (time.perf_counter() - t0) * 1000
                    if resp.status_code == 200:
                        result.get_latencies_ms.append(elapsed_ms)
                        bundle = resp.json()
                        entries = bundle.get("entry", [])
                        result.n_resources_found += len(entries)
                        url = None
                        for link in bundle.get("link", []):
                            if link.get("relation") == "next":
                                url = link.get("url")
                                break
                    else:
                        result.errors.append(f"GET {rtype}: HTTP {resp.status_code}")
                        break
                except Exception as exc:
                    result.errors.append(f"GET {rtype}: {exc}")
                    break
                page += 1

    log.info("GET benchmark: %d latências | stats: %s", len(result.get_latencies_ms), result.get_stats())
    return result


def benchmark_completeness(
    fhir_url: str,
    bundles_dir: Path,
    n_samples: int = 50,
) -> FHIRBenchmarkResult:
    """
    Verifica completeness: fração dos Patient IDs dos bundles que existem no HAPI FHIR.

    Um score < 1.0 indica que alguns bundles não foram carregados com sucesso.

    Args:
        fhir_url:    URL base do HAPI FHIR.
        bundles_dir: Diretório com bundle_NNNNNN.json.
        n_samples:   Número de bundles a verificar.

    Returns:
        FHIRBenchmarkResult com completeness_score.
    """
    import json as _json

    result = FHIRBenchmarkResult()
    bundle_files = sorted(bundles_dir.glob("bundle_*.json"))[:n_samples]

    if not bundle_files:
        return result

    result.n_resources_expected = len(bundle_files)
    base = fhir_url.rstrip("/")

    with httpx.Client(timeout=_REQUEST_TIMEOUT) as client:
        for bf in bundle_files:
            try:
                data = _json.loads(bf.read_text(encoding="utf-8"))
                # Extrair patient_id do primeiro entry Patient
                patient_id = None
                for entry in data.get("entry", []):
                    res = entry.get("resource", {})
                    if res.get("resourceType") == "Patient":
                        for ident in res.get("identifier", []):
                            patient_id = ident.get("value")
                            break
                        break

                if not patient_id:
                    continue

                resp = client.get(
                    f"{base}/Patient?identifier={patient_id}",
                    timeout=_REQUEST_TIMEOUT,
                )
                if resp.status_code == 200:
                    found = resp.json().get("total", 0)
                    if found > 0:
                        result.n_resources_found += 1
                    else:
                        result.errors.append(f"Patient {patient_id} não encontrado no FHIR.")
            except Exception as exc:
                result.errors.append(f"{bf.name}: {exc}")

    result.completeness_score = result.n_resources_found / max(result.n_resources_expected, 1)
    log.info(
        "Completeness: %.2f%% (%d/%d)",
        result.completeness_score * 100,
        result.n_resources_found,
        result.n_resources_expected,
    )
    return result


def run_full_benchmark(
    fhir_url: str = _DEFAULT_FHIR_URL,
    bundles_dir: str = "etl_worker/data/bundles",
    n_samples: int = 50,
) -> dict[str, dict]:
    """
    Executa o benchmark completo (POST + GET + completeness) e retorna resultados.

    Returns:
        {"post": {...}, "get": {...}, "completeness": {...}}
    """
    bdir = Path(bundles_dir)
    results = {}

    log.info("=== FHIR Benchmark — %s ===", fhir_url)

    post_result = benchmark_post_bundles(fhir_url, bdir, n_samples=n_samples)
    results["post"] = {
        **post_result.post_stats(),
        "n_bundles": post_result.n_bundles_posted,
        "n_errors":  len(post_result.errors),
    }

    get_result = benchmark_get_resources(fhir_url)
    results["get"] = {
        **get_result.get_stats(),
        "n_resources": get_result.n_resources_found,
        "n_errors":    len(get_result.errors),
    }

    comp_result = benchmark_completeness(fhir_url, bdir, n_samples=n_samples)
    results["completeness"] = {
        "score":        comp_result.completeness_score,
        "n_found":      comp_result.n_resources_found,
        "n_expected":   comp_result.n_resources_expected,
        "n_errors":     len(comp_result.errors),
    }

    log.info("\n%s", post_result)
    return results


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )
    parser = argparse.ArgumentParser(description="Benchmark de latência e completeness do HAPI FHIR.")
    parser.add_argument("--fhir-url", default=_DEFAULT_FHIR_URL)
    parser.add_argument("--bundles-dir", default="etl_worker/data/bundles")
    parser.add_argument("--n-samples", type=int, default=50)
    args = parser.parse_args()

    results = run_full_benchmark(
        fhir_url=args.fhir_url,
        bundles_dir=args.bundles_dir,
        n_samples=args.n_samples,
    )

    import json
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
