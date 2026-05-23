#!/usr/bin/env python3
"""
Minimal FHIR R4 in-memory server for federated learning experiments.

Implements exactly the endpoints used by fhir_consumer.py:
  GET  /fhir/metadata          - capability statement (readiness probe)
  POST /fhir                   - FHIR transaction bundle (ETL ingestion)
  GET  /fhir/Condition         - search (_count, _offset, subject)
  GET  /fhir/DocumentReference - search (_count, _offset, subject)
  GET  /fhir/Patient           - search (_count, _offset)
  GET  /fhir/Encounter         - search (_count, _offset, patient)

Usage:
    python fhir_server.py [port]   (default port: 8080)
"""

import json
import sys
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from urllib.parse import parse_qs, urlparse

_STORE: dict[str, dict[str, dict]] = {
    "Condition": {},
    "DocumentReference": {},
    "Patient": {},
    "Encounter": {},
}

_CAPABILITY = {
    "resourceType": "CapabilityStatement",
    "status": "active",
    "kind": "instance",
    "fhirVersion": "4.0.1",
    "format": ["application/fhir+json", "json"],
    "rest": [{"mode": "server"}],
}


def _searchset(resources: dict, count: int, offset: int, base_url: str) -> dict:
    items = list(resources.values())
    total = len(items)
    page = items[offset : offset + count]
    bundle: dict = {
        "resourceType": "Bundle",
        "type": "searchset",
        "total": total,
        "link": [{"relation": "self", "url": base_url}],
        "entry": [{"resource": r, "search": {"mode": "match"}} for r in page],
    }
    if offset + count < total:
        parsed = urlparse(base_url)
        next_url = (
            f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
            f"?_count={count}&_offset={offset + count}"
        )
        bundle["link"].append({"relation": "next", "url": next_url})
    return bundle


class _FHIRHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # suppress access logs
        pass

    def _json(self, data: dict, status: int = 200) -> None:
        body = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/fhir+json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _err(self, status: int, msg: str) -> None:
        self._json(
            {"resourceType": "OperationOutcome",
             "issue": [{"severity": "error", "diagnostics": msg}]},
            status,
        )

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        qs = parse_qs(parsed.query)

        def p(k: str, default: str) -> str:
            return qs.get(k, [default])[0]

        if path == "/fhir/metadata":
            self._json(_CAPABILITY)
            return

        parts = path.split("/fhir/", 1)
        if len(parts) < 2:
            self._err(404, "Not found")
            return

        rtype = parts[1]
        if rtype not in _STORE:
            self._err(404, f"Unknown resource type: {rtype}")
            return

        count = int(p("_count", "50"))
        offset = int(p("_offset", "0"))
        subject = p("subject", "") or p("patient", "")

        resources = _STORE[rtype]
        if subject:
            resources = {
                k: v for k, v in resources.items()
                if v.get("subject", {}).get("reference", "") == subject
                or v.get("patient", {}).get("reference", "") == subject
            }

        base_url = f"http://localhost:{PORT}{path}"
        self._json(_searchset(resources, count, offset, base_url))

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")

        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)

        try:
            bundle = json.loads(body)
        except json.JSONDecodeError:
            self._err(400, "Invalid JSON")
            return

        if path != "/fhir" or bundle.get("resourceType") != "Bundle":
            self._err(400, "Expected FHIR transaction Bundle at POST /fhir")
            return

        response_entries = []
        for entry in bundle.get("entry", []):
            resource = entry.get("resource", {})
            rtype = resource.get("resourceType", "")
            rid = resource.get("id") or str(uuid.uuid4())
            resource["id"] = rid

            if rtype in _STORE:
                _STORE[rtype][rid] = resource

            response_entries.append(
                {"response": {"status": "201 Created", "location": f"{rtype}/{rid}"}}
            )

        self._json(
            {"resourceType": "Bundle", "type": "transaction-response",
             "entry": response_entries},
            200,
        )


class _ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8080

if __name__ == "__main__":
    server = _ThreadedHTTPServer(("0.0.0.0", PORT), _FHIRHandler)
    print(f"FHIR server ready on http://0.0.0.0:{PORT}/fhir", flush=True)
    server.serve_forever()
