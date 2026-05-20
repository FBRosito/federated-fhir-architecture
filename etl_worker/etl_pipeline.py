"""
etl_pipeline.py
---------------
Lê evoluções clínicas em CSV particionadas de forma Non-IID e as converte para
recursos FHIR R5 (Patient, Condition, Composition, DocumentReference), enviando
um Transaction Bundle via POST para o servidor HAPI FHIR para validação.

Particionamento Non-IID:
  - Partição 0 (cardiology)      → predominância cardiovascular
  - Partição 1 (pneumology)      → predominância respiratória
  - Partição 2 (endocrinology)   → predominância metabólica/endócrina
  - Partição 3 (general)         → distribuição mista (UPA)

Uso:
    uv run python etl_worker/etl_pipeline.py [--partition <id>] [--data <path>]

Variáveis de ambiente:
    FHIR_SERVER_URL   URL base do HAPI FHIR (default: http://localhost:8080/fhir)
    ETL_PARTITION_ID  Partição a processar; -1 processa todas (default: -1)
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import pandas as pd
import httpx
from fhir.resources.bundle import Bundle, BundleEntry, BundleEntryRequest
from fhir.resources.composition import Composition
from fhir.resources.condition import Condition
from fhir.resources.documentreference import DocumentReference
from fhir.resources.patient import Patient

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("etl_pipeline")

# ── Constantes FHIR ───────────────────────────────────────────────────────────

LOINC_PROGRESS_NOTE = {"system": "http://loinc.org", "code": "11506-3", "display": "Progress note"}
LOINC_DISCHARGE_SUMMARY = {"system": "http://loinc.org", "code": "18842-5", "display": "Discharge summary"}
SNOMED_ENCOUNTER = {"system": "http://snomed.info/sct", "code": "371531000", "display": "Report of clinical encounter"}

COND_CLINICAL_ACTIVE = {
    "coding": [{"system": "http://terminology.hl7.org/CodeSystem/condition-clinical", "code": "active", "display": "Active"}]
}
COND_VER_CONFIRMED = {
    "coding": [{"system": "http://terminology.hl7.org/CodeSystem/condition-ver-status", "code": "confirmed", "display": "Confirmed"}]
}

# ── Mapeamento diagnóstico → CID-10 ──────────────────────────────────────────

ICD10_MAP: dict[str, tuple[str, str]] = {
    # Cardiovascular
    "Hipertensão arterial sistêmica":      ("I10",   "Hipertensão essencial (primária)"),
    "Insuficiência cardíaca congestiva":   ("I50.0",  "Insuficiência cardíaca congestiva"),
    "Angina instável":                     ("I20.0",  "Angina instável"),
    "Doença valvar aórtica":               ("I35.9",  "Transtorno da valva aórtica, não especificado"),
    "AVC isquêmico":                       ("I63.9",  "Infarto cerebral, não especificado"),
    "Infarto agudo do miocárdio":          ("I21.9",  "Infarto agudo do miocárdio, não especificado"),
    # Respiratório
    "DPOC exacerbado":                     ("J44.1",  "DPOC com exacerbação aguda"),
    "Asma brônquica":                      ("J45.9",  "Asma, não especificada"),
    "Pneumonia":                           ("J18.9",  "Pneumonia não especificada"),
    "Derrame pleural":                     ("J90",    "Derrame pleural não classificado em outra parte"),
    "Tromboembolismo pulmonar":            ("I26.9",  "Embolia pulmonar sem cor pulmonale agudo"),
    # Metabólico / Endócrino
    "Diabetes mellitus tipo 2":            ("E11.9",  "Diabetes mellitus tipo 2 sem complicações"),
    "Hipotireoidismo":                     ("E03.9",  "Hipotireoidismo, não especificado"),
    "Síndrome metabólica":                 ("E88.81", "Síndrome metabólica"),
    "Hiperparatireoidismo primário":       ("E21.0",  "Hiperparatireoidismo primário"),
    "Síndrome dos ovários policísticos":   ("E28.2",  "Síndrome dos ovários policísticos"),
    # Geral / Outros
    "Lombalgia aguda":                     ("M54.5",  "Dor lombar baixa"),
    "Infecção do trato urinário":          ("N39.0",  "Infecção do trato urinário, local não especificado"),
    "Artrite reumatoide":                  ("M05.9",  "Artrite reumatoide soropositiva, não especificada"),
    "Depressão":                           ("F32.9",  "Episódio depressivo, não especificado"),
    "Doença renal crônica":                ("N18.3",  "Doença renal crônica, estágio 3"),
}

# ── Carregamento de dados ─────────────────────────────────────────────────────

def load_data(data_path: Path, partition_id: int = -1) -> pd.DataFrame:
    """
    Carrega o CSV de evoluções clínicas.

    Args:
        data_path:    Caminho para o arquivo CSV.
        partition_id: Filtra por partição específica (-1 = todas).

    Returns:
        DataFrame com as evoluções do(s) partition(s) solicitado(s).
    """
    df = pd.read_csv(data_path, dtype={"partition_id": int})
    log.info("CSV carregado: %d registros em %d partições", len(df), df["partition_id"].nunique())

    if partition_id >= 0:
        df = df[df["partition_id"] == partition_id].copy()
        if df.empty:
            log.warning("Nenhum registro encontrado para partição %d.", partition_id)
        else:
            log.info("Partição %d (%s): %d registros", partition_id, df["partition_label"].iloc[0], len(df))

    return df

# ── Construtores de recursos FHIR ─────────────────────────────────────────────

def build_patient(row: pd.Series) -> tuple[str, Patient]:
    """
    Cria um recurso Patient a partir de uma linha do CSV.

    Returns:
        (urn_ref, Patient) onde urn_ref é o urn:uuid usado nas referências internas.
    """
    patient_uid = str(uuid.uuid4())
    urn = f"urn:uuid:{patient_uid}"

    name_parts = row["patient_name"].split()
    family = name_parts[-1] if len(name_parts) > 1 else name_parts[0]
    given = name_parts[:-1] if len(name_parts) > 1 else []

    patient = Patient.model_validate({
        "resourceType": "Patient",
        "id": patient_uid,
        "identifier": [{
            "system": "http://hospital.example.org/patients",
            "value": row["patient_id"],
        }],
        "name": [{"family": family, "given": given, "text": row["patient_name"]}],
        "gender": row["gender"],
        "birthDate": row["birth_date"],
    })
    return urn, patient


def build_condition(row: pd.Series, patient_urn: str) -> tuple[str, Condition]:
    """
    Mapeia o diagnóstico textual para CID-10 e cria um recurso Condition.

    Diagnósticos não mapeados recebem o código Z03.89 ("Sem diagnóstico definido")
    e são sinalizados com WARNING para revisão manual.
    """
    cond_uid = str(uuid.uuid4())
    urn = f"urn:uuid:{cond_uid}"
    raw = str(row["raw_diagnosis"]).strip()

    if raw in ICD10_MAP:
        icd_code, icd_display = ICD10_MAP[raw]
    else:
        log.warning("Diagnóstico não mapeado: '%s'. Usando Z03.89.", raw)
        icd_code, icd_display = "Z03.89", "Sem diagnóstico relevante relevado"

    condition = Condition.model_validate({
        "resourceType": "Condition",
        "id": cond_uid,
        "clinicalStatus": COND_CLINICAL_ACTIVE,
        "verificationStatus": COND_VER_CONFIRMED,
        "category": [{
            "coding": [{
                "system": "http://terminology.hl7.org/CodeSystem/condition-category",
                "code": "encounter-diagnosis",
                "display": "Encounter Diagnosis",
            }]
        }],
        "code": {
            "coding": [{
                "system": "http://hl7.org/fhir/sid/icd-10",
                "code": icd_code,
                "display": icd_display,
            }],
            "text": raw,
        },
        "subject": {"reference": patient_urn},
        "recordedDate": row["record_date"],
        "note": [{"text": f"Diagnóstico extraído de evolução clínica — partição {row['partition_label']}"}],
    })
    return urn, condition


def build_composition(
    row: pd.Series,
    patient_urn: str,
    condition_urn: str,
) -> tuple[str, Composition]:
    """
    Cria um recurso Composition (nota de progresso clínico) referenciando o
    Patient e a Condition já construídos.
    """
    comp_uid = str(uuid.uuid4())
    urn = f"urn:uuid:{comp_uid}"
    practitioner_display = str(row.get("practitioner", "Profissional não identificado"))

    # Texto da seção formatado como XHTML mínimo (exigido pelo FHIR Narrative)
    xhtml_text = (
        f'<div xmlns="http://www.w3.org/1999/xhtml">'
        f"<p><b>Evolução:</b> {row['clinical_text']}</p>"
        f"<p><b>Diagnóstico:</b> {row['raw_diagnosis']}</p>"
        f"</div>"
    )

    composition = Composition.model_validate({
        "resourceType": "Composition",
        "id": comp_uid,
        "status": "final",
        "type": {"coding": [LOINC_PROGRESS_NOTE]},
        "subject": [{"reference": patient_urn}],
        "date": row["record_date"],
        "author": [{"display": practitioner_display}],
        "title": f"Evolução Clínica — {row['partition_label'].capitalize()}",
        "section": [{
            "title": "Evolução e Conduta",
            "code": {"coding": [LOINC_PROGRESS_NOTE]},
            "text": {"status": "generated", "div": xhtml_text},
            "entry": [{"reference": condition_urn}],
        }],
    })
    return urn, composition


def build_document_reference(
    row: pd.Series,
    patient_urn: str,
    composition_urn: str,
) -> tuple[str, DocumentReference]:
    """
    Cria um DocumentReference apontando para a Composition correspondente.
    O texto clínico é embutido como attachment base64 (text/plain).
    """
    doc_uid = str(uuid.uuid4())
    urn = f"urn:uuid:{doc_uid}"

    encoded_text = base64.b64encode(row["clinical_text"].encode("utf-8")).decode("ascii")

    doc_ref = DocumentReference.model_validate({
        "resourceType": "DocumentReference",
        "id": doc_uid,
        "status": "current",
        "docStatus": "final",
        "type": {"coding": [LOINC_PROGRESS_NOTE]},
        "category": [{"coding": [SNOMED_ENCOUNTER]}],
        "subject": {"reference": patient_urn},
        "date": row["record_date"],
        "author": [{"display": str(row.get("practitioner", ""))}],
        "description": f"Evolução clínica — {row['patient_name']} — {row['record_date'][:10]}",
        "content": [{
            "attachment": {
                "contentType": "text/plain;charset=UTF-8",
                "data": encoded_text,
                "title": f"Evolução {row['record_date'][:10]}",
                "creation": row["record_date"],
            }
        }],
        # relatesTo.target exige referência a outro DocumentReference (FHIR R4 §10.3.2).
        # A ligação com a Composition é mantida via Composition.section.entry (build_composition).

    })
    return urn, doc_ref

# ── Montagem do Bundle de transação ──────────────────────────────────────────

def _entry(urn: str, resource: Any) -> dict[str, Any]:
    """Monta uma entrada de Transaction Bundle a partir de um recurso FHIR."""
    return {
        "fullUrl": urn,
        "resource": json.loads(resource.model_dump_json(exclude_none=True)),
        "request": {
            "method": "POST",
            "url": resource.__resource_type__,
        },
    }


def build_transaction_bundle(
    patient: Patient, patient_urn: str,
    condition: Condition, condition_urn: str,
    composition: Composition, composition_urn: str,
    doc_ref: DocumentReference, doc_ref_urn: str,
) -> Bundle:
    """Agrupa os quatro recursos em um único Transaction Bundle."""
    return Bundle.model_validate({
        "resourceType": "Bundle",
        "type": "transaction",
        "entry": [
            _entry(patient_urn, patient),
            _entry(condition_urn, condition),
            _entry(composition_urn, composition),
            _entry(doc_ref_urn, doc_ref),
        ],
    })

# ── Envio para HAPI FHIR ──────────────────────────────────────────────────────

def post_bundle(
    bundle: Bundle,
    fhir_url: str,
    *,
    retry_interval: float = 5.0,
    timeout_total: float = 120.0,
) -> dict[str, Any]:
    """
    Envia um Transaction Bundle via POST para o endpoint FHIR com retentativas.

    Erros transitórios (servidor ainda inicializando) são retentados a cada
    `retry_interval` segundos por até `timeout_total` segundos antes de desistir.
    Erros permanentes de cliente (4xx) não são retentados.

    Args:
        bundle:        Bundle FHIR a enviar.
        fhir_url:      URL base do servidor HAPI FHIR.
        retry_interval: Intervalo em segundos entre tentativas (padrão: 5 s).
        timeout_total:  Tempo máximo de espera acumulado em segundos (padrão: 120 s).

    Returns:
        Dicionário com a resposta do servidor (Bundle de resposta ou OperationOutcome).

    Raises:
        httpx.HTTPStatusError: quando o servidor retorna 4xx (erro permanente).
        httpx.TransportError:  quando o servidor permanece inacessível após o timeout total.
    """
    payload = bundle.model_dump_json(exclude_none=True).encode("utf-8")
    headers = {"Content-Type": "application/fhir+json; charset=UTF-8"}
    deadline = time.monotonic() + timeout_total
    attempt = 0

    while True:
        attempt += 1
        try:
            response = httpx.post(fhir_url, content=payload, headers=headers, timeout=30)

            content_type = response.headers.get("content-type", "")
            is_fhir_json = "json" in content_type or "fhir" in content_type

            if 200 <= response.status_code < 300:
                return response.json()

            # 4xx com JSON/FHIR → erro real do cliente (bundle inválido); não retentar
            if 400 <= response.status_code < 500 and is_fhir_json:
                response.raise_for_status()

            # 4xx com HTML = Tomcat ainda inicializando (boot do JPA incompleto);
            # 5xx = servidor sobrecarregado/reiniciando — ambos são transitórios
            raise httpx.HTTPStatusError(
                f"Servidor retornou {response.status_code} (Content-Type: {content_type!r})",
                request=response.request,
                response=response,
            )

        except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError) as exc:
            remaining = deadline - time.monotonic()
            is_permanent_client_error = (
                isinstance(exc, httpx.HTTPStatusError)
                and exc.response.status_code < 500
                and ("json" in exc.response.headers.get("content-type", "")
                     or "fhir" in exc.response.headers.get("content-type", ""))
            )
            if is_permanent_client_error or remaining <= 0:
                raise

            log.warning(
                "Tentativa %d — HAPI FHIR indisponível (%s). "
                "Nova tentativa em %.0f s (restam %.0f s).",
                attempt, exc, retry_interval, remaining,
            )
            time.sleep(min(retry_interval, remaining))


def _summarise_response(resp: dict[str, Any], patient_id: str) -> None:
    """Loga o resultado de cada entrada do Bundle de resposta."""
    for entry in resp.get("entry", []):
        req = entry.get("response", {})
        status = req.get("status", "?")
        location = req.get("location", "—")
        outcome = req.get("outcome", {})
        issues = outcome.get("issue", []) if outcome else []
        severity = issues[0].get("severity", "") if issues else ""
        if severity in ("error", "fatal"):
            log.error("[%s] %s → %s  %s", patient_id, status, location, issues)
        else:
            log.info("[%s] %s → %s", patient_id, status, location)

# ── Pipeline principal ────────────────────────────────────────────────────────

def process_row(row: pd.Series, fhir_url: str, dry_run: bool = False) -> bool:
    """
    Processa uma única linha do CSV: constrói os recursos, monta o Bundle e envia.

    Args:
        row:      Linha do DataFrame pandas.
        fhir_url: URL base do HAPI FHIR (ex.: http://localhost:8080/fhir).
        dry_run:  Se True, serializa o Bundle mas não o envia.

    Returns:
        True em caso de sucesso, False em caso de erro.
    """
    try:
        patient_urn,   patient    = build_patient(row)
        condition_urn, condition  = build_condition(row, patient_urn)
        comp_urn,      composition = build_composition(row, patient_urn, condition_urn)
        doc_urn,       doc_ref    = build_document_reference(row, patient_urn, comp_urn)

        bundle = build_transaction_bundle(
            patient, patient_urn,
            condition, condition_urn,
            composition, comp_urn,
            doc_ref, doc_urn,
        )

        if dry_run:
            log.info("[DRY-RUN] Bundle para %s — %d entries, %d bytes",
                     row["patient_id"], len(bundle.entry),
                     len(bundle.model_dump_json(exclude_none=True)))
            return True

        resp = post_bundle(bundle, fhir_url)
        _summarise_response(resp, row["patient_id"])
        return True

    except httpx.HTTPStatusError as exc:
        log.error("HTTP %s ao processar %s: %s",
                  exc.response.status_code, row["patient_id"], exc.response.text[:300])
    except httpx.ConnectError:
        log.error("Sem conexão com FHIR em %s. Verifique se o container hapi_fhir está no ar.", fhir_url)
    except Exception as exc:  # noqa: BLE001
        log.exception("Erro inesperado ao processar %s: %s", row["patient_id"], exc)

    return False


def run(
    data_path: Path,
    fhir_url: str,
    partition_id: int = -1,
    dry_run: bool = False,
) -> None:
    """
    Executa o pipeline ETL completo para o(s) partition(s) solicitado(s).

    Args:
        data_path:    Caminho para o CSV de evoluções clínicas.
        fhir_url:     URL base do servidor HAPI FHIR.
        partition_id: Partição a processar (-1 = todas).
        dry_run:      Valida sem enviar ao servidor.
    """
    df = load_data(data_path, partition_id)
    if df.empty:
        log.warning("Nenhum dado para processar. Encerrando.")
        return

    success = errors = 0
    for _, row in df.iterrows():
        if process_row(row, fhir_url, dry_run=dry_run):
            success += 1
        else:
            errors += 1

    total = success + errors
    log.info(
        "Pipeline concluído: %d/%d registros enviados com sucesso%s.",
        success, total,
        " [DRY-RUN]" if dry_run else "",
    )
    if errors:
        log.warning("%d registro(s) com falha — verifique os logs acima.", errors)
        sys.exit(1)

# ── Ponto de entrada CLI ──────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="ETL: evoluções clínicas CSV → FHIR Transaction Bundle → HAPI FHIR",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).parent / "data" / "clinical_evolutions.csv",
        help="Caminho para o CSV de evoluções clínicas.",
    )
    parser.add_argument(
        "--partition",
        type=int,
        default=int(os.getenv("ETL_PARTITION_ID", "-1")),
        help="ID da partição a processar (-1 = todas).",
    )
    parser.add_argument(
        "--fhir-url",
        default=os.getenv("FHIR_SERVER_URL", "http://localhost:8080/fhir"),
        help="URL base do servidor HAPI FHIR.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Valida e serializa os Bundles sem enviá-los ao servidor.",
    )
    args = parser.parse_args()

    log.info("=== ETL Pipeline iniciado ===")
    log.info("Dados: %s | FHIR: %s | Partição: %s | Dry-run: %s",
             args.data, args.fhir_url,
             args.partition if args.partition >= 0 else "todas",
             args.dry_run)

    run(
        data_path=args.data,
        fhir_url=args.fhir_url,
        partition_id=args.partition,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
