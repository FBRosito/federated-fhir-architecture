"""
cross_silo_eval.py
-------------------
Article 3, Part 4: cross-silo generalization eval for dual_lora.

For each trained silo k, reloads its final aggregated GLOBAL adapter
(final_global_params.npz, saved by client.py's fit() at the last round) and
its LOCAL adapter checkpoint (local_adapter.pt), activates ONLY the global
adapter, and evaluates on every OTHER silo j > k's held-out eval split.
Answers: does the DP-trained global component, once aggregated, generalize
to a distribution it never saw during training — or did federation just
average together silo-overfit adapters?

The local adapter is loaded (per the spec) even though only "default" is
active for this eval, since get_peft_model_state_dict/set_peft_model_state_dict
are adapter-name-scoped (see ai_client.model_setup_bert.get_bert_parameters) —
loading it costs nothing and keeps the reconstructed model identical in
structure to what fit()/evaluate() actually ran with.

Usage:
    uv run python -m article3.cross_silo_eval --tag a3_dual_lora_sigma1.0_seed0
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from peft import set_peft_model_state_dict
from torch.utils.data import DataLoader

from ai_client.fhir_consumer import fetch_training_examples
from ai_client.fl_client import _stratified_split
from ai_client.model_setup_bert import build_bert_dataset, load_bert_model, set_bert_parameters
from evaluation.icd_metrics import evaluate_bert_model


def _load_silo_eval_examples(fhir_url: str, partition_id: int) -> list:
    examples, stats = fetch_training_examples(fhir_url)
    for w in stats.warnings:
        print(f"  FHIR consumer warning: {w}")
    tag = f"partition_id={partition_id}"
    examples = [ex for ex in examples if tag in ex.partition_note or not ex.partition_note]
    _train_ex, eval_ex = _stratified_split(examples)
    return eval_ex


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True, help="Experiment tag, e.g. a3_dual_lora_sigma1.0_seed0")
    parser.add_argument("--logs-dir", default="logs")
    parser.add_argument("--fhir-url", default=os.getenv("FHIR_SERVER_URL", "http://localhost:8080/fhir"))
    parser.add_argument("--num-silos", type=int, default=5)
    parser.add_argument("--label-index-path", default=os.getenv("BERT_LABEL_INDEX_PATH", ""))
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    logs_root = Path(args.logs_dir)
    tag_dir = logs_root / args.tag
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.label_index_path) as f:
        full_idx = json.load(f)
    label_index = {k: v for k, v in full_idx.items() if v < 50}  # top50 benchmark, matches run_dual_lora.sh

    print(f"Fetching eval examples for {args.num_silos} silos from {args.fhir_url}...")
    silo_eval_examples = {
        j: _load_silo_eval_examples(args.fhir_url, j) for j in range(args.num_silos)
    }
    for j, exs in silo_eval_examples.items():
        print(f"  silo {j}: {len(exs)} eval examples")

    out_path = logs_root / "cross_silo_eval.jsonl"
    with open(out_path, "a") as out_f:
        for k in range(args.num_silos):
            silo_dir = tag_dir / str(k)
            global_ckpt = silo_dir / "final_global_params.npz"
            local_ckpt = silo_dir / "local_adapter.pt"

            if not global_ckpt.exists():
                print(f"MISSING checkpoint for silo {k}: {global_ckpt} — skipping.")
                continue

            model, tokenizer = load_bert_model(num_labels=50)
            npz = np.load(global_ckpt)
            params = [npz[f"arr_{i}"] for i in range(len(npz.files))]
            set_bert_parameters(model, params)

            if local_ckpt.exists():
                local_state = torch.load(local_ckpt, map_location="cpu")
                set_peft_model_state_dict(model.encoder, local_state, adapter_name="local")
            else:
                print(f"  (no local_adapter.pt for silo {k} — evaluating global-only anyway)")

            model.encoder.set_adapter("default")  # only the global component, per spec
            model = model.to(device)

            for j in range(k + 1, args.num_silos):
                eval_ds = build_bert_dataset(
                    silo_eval_examples[j], label_index, tokenizer, args.max_length, num_labels=50,
                )
                eval_dl = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False)
                result = evaluate_bert_model(model, eval_dl, device)

                record = {
                    "tag": args.tag,
                    "train_silo": k,
                    "eval_silo": j,
                    "micro_f1": result.micro_f1,
                    "n_examples": len(eval_ds),
                }
                out_f.write(json.dumps(record) + "\n")
                out_f.flush()
                print(f"  silo {k} -> silo {j}: micro_f1={result.micro_f1:.4f} (n={len(eval_ds)})")

            del model
            torch.cuda.empty_cache()

    print(f"\nWrote cross-silo eval records to {out_path}")


if __name__ == "__main__":
    main()
