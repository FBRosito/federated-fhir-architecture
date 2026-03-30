"""
fhir_consumer.py
----------------
Consulta o servidor HAPI FHIR para recuperar recursos Condition (rótulos CID-10)
e DocumentReference (textos clínicos) e os combina em exemplos de treinamento
prontos para o pipeline de fine-tuning federado.

Variáveis de ambiente:
    FHIR_SERVER_URL   URL base do HAPI FHIR (default: http://localhost:8080/fhir)

Uso direto:
    uv run python ai_client/fhir_consumer.py [--fhir-url URL] [--patient-id ID]
"""

from __future__ import annotations

import argparse
import base64
import logging
import os
from dataclasses import dataclass, field
from typing import Any

import requests

log = logging.getLogger(__name__)

# ── Constantes ────────────────────────────────────────────────────────────────

ICD10_SYSTEM      = "http://hl7.org/fhir/sid/icd-10"
LOINC_SYSTEM      = "http://loinc.org"
_DEFAULT_PAGE_SIZE = 50          # _count por página nas buscas FHIR
_REQUEST_TIMEOUT   = 30          # segundos

# ── Modelo de dados ───────────────────────────────────────────────────────────

@dataclass
class TrainingExample:
    """
    Par (texto clínico, rótulo CID-10) extraído do servidor FHIR.

    Attributes:
        patient_ref:    Referência canônica ao Patient (ex.: "Patient/uuid").
        clinical_text:  Texto da evolução clínica decodificado de base64.
        icd10_code:     Código CID-10 da Condition vinculada ao paciente.
        icd10_display:  Descrição textual do código CID-10.
        condition_id:   ID do recurso Condition no servidor.
        doc_ref_id:     ID do recurso DocumentReference no servidor.
        partition_note: Nota de partição Non-IID extraída do campo note, se presente.
    """
    patient_ref:    str
    clinical_text:  str
    icd10_code:     str
    icd10_display:  str
    condition_id:   str = ""
    doc_ref_id:     str = ""
    partition_note: str = ""

    def to_prompt(self) -> str:
        """
        Formata o exemplo no template de instrução usado pelo fine-tuning
        (Alpaca-style, compatível com Llama-3).
        """
        return (
            "### Instrução:\n"
            "Analise a evolução clínica abaixo e identifique o código CID-10 correspondente.\n\n"
            "### Evolução Clínica:\n"
            f"{self.clinical_text.strip()}\n\n"
            "### Resposta:\n"
            f"{self.icd10_code} — {self.icd10_display}"
        )


@dataclass
class FHIRConsumerStats:
    """Estatísticas da última execução de `fetch_training_examples`."""
    conditions_fetched:      int = 0
    doc_refs_fetched:        int = 0
    examples_paired:         int = 0
    conditions_no_icd10:     int = 0
    doc_refs_no_text:        int = 0
    patients_with_both:      int = 0
    warnings:                list[str] = field(default_factory=list)


# ── Paginação FHIR ────────────────────────────────────────────────────────────

def _next_link(bundle: dict[str, Any]) -> str | None:
    """Extrai a URL da próxima página de um Bundle searchset, ou None."""
    for link in bundle.get("link", []):
        if link.get("relation") == "next":
            return link.get("url")
    return None


def _iter_bundle_entries(
    session: requests.Session,
    url: str,
    params: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Percorre todas as páginas de um searchset FHIR e devolve todos os entries.

    Args:
        session: Sessão HTTP reutilizável.
        url:     Endpoint de busca (ex.: "{base}/Condition").
        params:  Query-string da busca (_count, _fields, subject, etc.).

    Returns:
        Lista com todos os dicts `entry[].resource` encontrados.
    """
    resources: list[dict[str, Any]] = []
    next_url: str | None = url
    first_call = True

    while next_url:
        resp = session.get(
            next_url,
            params=params if first_call else None,
            timeout=_REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        bundle = resp.json()

        for entry in bundle.get("entry", []):
            resource = entry.get("resource")
            if resource:
                resources.append(resource)

        next_url = _next_link(bundle)
        first_call = False

    return resources


# ── Extração de Conditions ────────────────────────────────────────────────────

def _extract_icd10(condition: dict[str, Any]) -> tuple[str, str] | None:
    """
    Extrai (code, display) ICD-10 do coding array de uma Condition.

    Retorna None se não houver coding com o sistema ICD-10.
    """
    code_cc = condition.get("code", {})
    for coding in code_cc.get("coding", []):
        if coding.get("system") == ICD10_SYSTEM:
            return coding.get("code", ""), coding.get("display", "")
    return None


def get_conditions(
    fhir_url: str,
    patient_id: str | None = None,
    session: requests.Session | None = None,
) -> dict[str, tuple[str, str, str]]:
    """
    Recupera recursos Condition do servidor HAPI FHIR.

    Args:
        fhir_url:   URL base do servidor (ex.: "http://localhost:8080/fhir").
        patient_id: Se fornecido, filtra apenas as Conditions do paciente.
        session:    Sessão HTTP a reutilizar (criada internamente se None).

    Returns:
        Dicionário `{patient_ref: (condition_id, icd10_code, icd10_display)}`.
        Apenas condições com código ICD-10 válido são incluídas.
        Quando múltiplas Conditions existem para o mesmo paciente, a primeira
        (por ordem de chegada) prevalece — adequado para o cenário one-label FL.
    """
    sess = session or requests.Session()
    params: dict[str, Any] = {"_count": _DEFAULT_PAGE_SIZE}
    if patient_id:
        params["subject"] = f"Patient/{patient_id}"

    endpoint = f"{fhir_url.rstrip('/')}/Condition"
    log.info("Buscando Conditions em %s (patient_id=%s)…", endpoint, patient_id or "todos")

    raw = _iter_bundle_entries(sess, endpoint, params)
    log.info("  → %d Condition(s) retornada(s)", len(raw))

    result: dict[str, tuple[str, str, str]] = {}
    for cond in raw:
        subj = cond.get("subject", {}).get("reference", "")
        if not subj:
            continue
        if subj in result:
            continue  # já temos um CID-10 para este paciente

        icd = _extract_icd10(cond)
        if icd is None:
            log.debug("Condition %s sem CID-10 — ignorada.", cond.get("id", "?"))
            continue

        result[subj] = (cond.get("id", ""), icd[0], icd[1])

    return result


# ── Extração de DocumentReferences ───────────────────────────────────────────

def _decode_attachment(content_list: list[dict]) -> str | None:
    """
    Decodifica o primeiro attachment base64 text/* de um DocumentReference.

    Returns:
        Texto decodificado ou None se não houver attachment com dados.
    """
    for content_item in content_list:
        attachment = content_item.get("attachment", {})
        data_b64 = attachment.get("data")
        if data_b64:
            try:
                return base64.b64decode(data_b64).decode("utf-8", errors="replace")
            except Exception as exc:
                log.warning("Falha ao decodificar attachment: %s", exc)
    return None


def get_document_references(
    fhir_url: str,
    patient_id: str | None = None,
    session: requests.Session | None = None,
) -> dict[str, tuple[str, str]]:
    """
    Recupera recursos DocumentReference do servidor HAPI FHIR.

    Args:
        fhir_url:   URL base do servidor.
        patient_id: Filtra por paciente específico (opcional).
        session:    Sessão HTTP reutilizável.

    Returns:
        Dicionário `{patient_ref: (doc_ref_id, clinical_text)}`.
        Apenas DocumentReferences com attachment text/* decodificável são incluídos.
        Quando múltiplos documentos existem para o mesmo paciente, o primeiro prevalece.
    """
    sess = session or requests.Session()
    params: dict[str, Any] = {"_count": _DEFAULT_PAGE_SIZE}
    if patient_id:
        params["subject"] = f"Patient/{patient_id}"

    endpoint = f"{fhir_url.rstrip('/')}/DocumentReference"
    log.info("Buscando DocumentReferences em %s (patient_id=%s)…", endpoint, patient_id or "todos")

    raw = _iter_bundle_entries(sess, endpoint, params)
    log.info("  → %d DocumentReference(s) retornada(s)", len(raw))

    result: dict[str, tuple[str, str]] = {}
    for doc in raw:
        subj = doc.get("subject", {}).get("reference", "")
        if not subj:
            continue
        if subj in result:
            continue

        text = _decode_attachment(doc.get("content", []))
        if text is None:
            log.debug("DocumentReference %s sem attachment decodificável — ignorado.", doc.get("id", "?"))
            continue

        result[subj] = (doc.get("id", ""), text)

    return result


# ── Combinação em exemplos de treinamento ─────────────────────────────────────

def fetch_training_examples(
    fhir_url: str,
    patient_id: str | None = None,
    min_text_length: int = 20,
) -> tuple[list[TrainingExample], FHIRConsumerStats]:
    """
    Ponto de entrada principal: combina Conditions e DocumentReferences
    pelo campo `subject` (patient reference) e retorna exemplos de treinamento.

    Um exemplo é gerado apenas quando **ambos** os recursos estão disponíveis
    para o mesmo paciente e o texto clínico tem comprimento mínimo razoável.

    Args:
        fhir_url:        URL base do servidor HAPI FHIR.
        patient_id:      Filtra por paciente específico (opcional).
        min_text_length: Comprimento mínimo (chars) do texto clínico para inclusão.

    Returns:
        (exemplos, stats) onde `stats` reporta contagens e avisos da execução.
    """
    stats = FHIRConsumerStats()
    examples: list[TrainingExample] = []

    with requests.Session() as session:
        session.headers.update({
            "Accept": "application/fhir+json",
            "Content-Type": "application/fhir+json",
        })

        try:
            conditions  = get_conditions(fhir_url, patient_id, session)
            doc_refs    = get_document_references(fhir_url, patient_id, session)
        except requests.ConnectionError:
            msg = f"Sem conexão com FHIR em {fhir_url}. Verifique se o container hapi_fhir está no ar."
            log.error(msg)
            stats.warnings.append(msg)
            return [], stats
        except requests.HTTPError as exc:
            msg = f"HTTP {exc.response.status_code} ao consultar FHIR: {exc.response.text[:200]}"
            log.error(msg)
            stats.warnings.append(msg)
            return [], stats

    stats.conditions_fetched = len(conditions)
    stats.doc_refs_fetched   = len(doc_refs)

    # Diagnóstico de cobertura
    only_conditions  = set(conditions) - set(doc_refs)
    only_doc_refs    = set(doc_refs)   - set(conditions)
    both             = set(conditions) & set(doc_refs)

    stats.patients_with_both    = len(both)
    stats.conditions_no_icd10   = 0   # já filtrado em get_conditions
    stats.doc_refs_no_text      = 0   # já filtrado em get_document_references

    if only_conditions:
        w = f"{len(only_conditions)} paciente(s) com Condition mas sem DocumentReference: {sorted(only_conditions)[:5]}…"
        log.warning(w)
        stats.warnings.append(w)
    if only_doc_refs:
        w = f"{len(only_doc_refs)} paciente(s) com DocumentReference mas sem Condition: {sorted(only_doc_refs)[:5]}…"
        log.warning(w)
        stats.warnings.append(w)

    # Montagem dos exemplos
    for patient_ref in sorted(both):
        cond_id, icd10_code, icd10_display = conditions[patient_ref]
        doc_id,  clinical_text             = doc_refs[patient_ref]

        if len(clinical_text.strip()) < min_text_length:
            log.debug("Texto muito curto para %s (%d chars) — ignorado.", patient_ref, len(clinical_text))
            continue

        examples.append(TrainingExample(
            patient_ref    = patient_ref,
            clinical_text  = clinical_text,
            icd10_code     = icd10_code,
            icd10_display  = icd10_display,
            condition_id   = cond_id,
            doc_ref_id     = doc_id,
        ))

    stats.examples_paired = len(examples)
    log.info(
        "Pareamento concluído: %d/%d pacientes com ambos os recursos → %d exemplos de treinamento.",
        len(both), max(len(conditions), len(doc_refs), 1), len(examples),
    )
    return examples, stats


# ── CLI de diagnóstico ────────────────────────────────────────────────────────

def _cli() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Diagnóstico do consumer FHIR — lista exemplos disponíveis para treinamento.",
    )
    parser.add_argument(
        "--fhir-url",
        default=os.getenv("FHIR_SERVER_URL", "http://localhost:8080/fhir"),
        help="URL base do servidor HAPI FHIR.",
    )
    parser.add_argument("--patient-id", default=None, help="Filtra por Patient.id.")
    parser.add_argument("--show-prompts", action="store_true", help="Exibe os prompts formatados.")
    args = parser.parse_args()

    examples, stats = fetch_training_examples(args.fhir_url, args.patient_id)

    print(f"\n{'─'*60}")
    print(f"  Conditions recuperadas : {stats.conditions_fetched}")
    print(f"  DocumentRefs recup.    : {stats.doc_refs_fetched}")
    print(f"  Pacientes com ambos    : {stats.patients_with_both}")
    print(f"  Exemplos de treino     : {stats.examples_paired}")
    if stats.warnings:
        print(f"  Avisos                 : {len(stats.warnings)}")
        for w in stats.warnings:
            print(f"    ⚠ {w}")
    print(f"{'─'*60}\n")

    if args.show_prompts:
        for i, ex in enumerate(examples, 1):
            print(f"[{i}] patient={ex.patient_ref}")
            print(ex.to_prompt())
            print()


if __name__ == "__main__":
    _cli()
