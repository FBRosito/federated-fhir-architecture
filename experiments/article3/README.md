# Article 3 — DP-LoRA / FFA-LoRA / Dual-LoRA (HERALD-PFL)

Standalone experiment package (own `.venv`, not part of the root uv workspace),
following the same structural pattern as `experiments/adaptive-clipping`.

- `dp_lora` / `ffa_lora`: reuse `experiments/adaptive-clipping`'s
  `adaptive-clipping-server`/`-client` binaries with `ADAPTIVE_CLIPPING_STRATEGY=baseline`
  and `FL_LORA_MODE=standard|ffa` — no new client logic needed for those two.
- `dual_lora`: needs its own client (adapter switching + local-adapter
  persistence across rounds) and server (round-JSONL logging), provided by
  this package's `dual-lora-client` / `dual-lora-server` entry points.

See `scripts/run_dual_lora.sh` for the run invocation and
`src/article3/dual_training.py` for the training-loop implementation.
