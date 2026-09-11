# Security Policy

## Scope

This repository contains research software for privacy-preserving federated
learning over clinical data (HL7 FHIR + DP-SGD). It must **never** contain:

- Real or synthetic patient-level data (MIMIC-IV CSVs, FHIR bundles, parquet, etc.)
- Credentials of any kind (API keys, HF tokens, TLS private keys, `.env` files)
- Trained model checkpoints that could memorize training examples

## Reporting a Vulnerability

If you discover a security or privacy issue — **including accidentally committed
sensitive data or credentials** — please report it privately:

- **Email:** security@ufcspa.edu.br
- **Subject:** `[HERALD] Security report`

Please do **not** open a public GitHub issue for sensitive reports.

Include: affected file(s)/commit(s), a description of the issue, and (if known)
whether the material has been pushed to a public remote. We will acknowledge
receipt within 5 business days.

## If you find committed secrets or patient data

1. Do not clone, fork, or redistribute the repository further.
2. Report it immediately via the email above.
3. Maintainers will rotate any exposed credentials and rewrite history
   (`git filter-repo`) to purge the material.

## Hardening notes for deployers

- MIMIC-IV must be obtained via PhysioNet credentialed access by each user;
  the ETL processes local copies only (`physionet.org/` is gitignored).
- Set `FL_NETWORK_MODE=real` and provide TLS certificates for any
  non-loopback deployment; the default `simulated` mode is insecure gRPC on
  localhost (see `.env.example`).
