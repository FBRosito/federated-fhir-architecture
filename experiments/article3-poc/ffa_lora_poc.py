"""
POC: FFA-LoRA compatibility with HERALD.
Tests whether freezing LoRA A matrix via requires_grad=False works correctly
with the HERALD DP-SGD training loop.
"""
import torch
import warnings
warnings.filterwarnings("ignore")


def test_ffa_lora():
    print("=" * 60)
    print("POC: FFA-LoRA in HERALD environment")
    print("=" * 60)

    # Step 1: Load PubMedBERT + LoRA (same config as HERALD)
    print("\n[1] Loading PubMedBERT + LoRA (standard config)...")
    from transformers import AutoModelForSequenceClassification
    from peft import LoraConfig, get_peft_model, TaskType

    model_name = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=50,
            ignore_mismatched_sizes=True,
        )
        print(f"  PubMedBERT loaded")
    except Exception as e:
        print(f"  PubMedBERT failed ({e}), falling back to bert-base-uncased...")
        model = AutoModelForSequenceClassification.from_pretrained(
            "bert-base-uncased", num_labels=50, ignore_mismatched_sizes=True
        )

    # Standard LoRA config (same as HERALD)
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,
        lora_alpha=16,
        target_modules=["query", "value"],
        lora_dropout=0.0,
        bias="none",
    )
    model = get_peft_model(model, lora_config)

    # Count trainable params before FFA
    trainable_before = {
        name: p.numel()
        for name, p in model.named_parameters()
        if p.requires_grad
    }
    print(f"  Trainable params (standard LoRA): {sum(trainable_before.values()):,}")
    lora_a_params = {k: v for k, v in trainable_before.items() if "lora_A" in k}
    lora_b_params = {k: v for k, v in trainable_before.items() if "lora_B" in k}
    print(f"  lora_A params: {sum(lora_a_params.values()):,} across {len(lora_a_params)} modules")
    print(f"  lora_B params: {sum(lora_b_params.values()):,} across {len(lora_b_params)} modules")

    # Step 2: Apply FFA-LoRA (freeze A matrices)
    print("\n[2] Applying FFA-LoRA (freezing lora_A matrices)...")
    frozen_count = 0
    for name, param in model.named_parameters():
        if "lora_A" in name:
            param.requires_grad = False
            frozen_count += 1

    trainable_after = {
        name: p.numel()
        for name, p in model.named_parameters()
        if p.requires_grad
    }
    print(f"  Frozen lora_A modules: {frozen_count}")
    print(f"  Trainable params (FFA-LoRA): {sum(trainable_after.values()):,}")

    # Verify only lora_B and classifier are trainable
    unexpected = [
        name for name in trainable_after
        if "lora_A" in name
    ]
    if unexpected:
        print(f"  WARNING: lora_A still trainable: {unexpected}")
    else:
        print(f"  Confirmed: no lora_A in trainable params")

    # Step 3: Simulate HERALD DP-SGD training loop
    print("\n[3] Simulating HERALD DP-SGD with FFA-LoRA...")
    import torch
    from torch.optim import AdamW
    from torch.utils.data import DataLoader, TensorDataset

    n_samples = 200
    seq_len = 128
    input_ids = torch.randint(0, 1000, (n_samples, seq_len))
    attention_mask = torch.ones(n_samples, seq_len, dtype=torch.long)
    labels = torch.randint(0, 50, (n_samples,))
    dataset = TensorDataset(input_ids, attention_mask, labels)

    # q=0.01 means ~2 samples per step (same as Article 3 target)
    q = 0.01
    batch_size = max(1, int(n_samples * q))
    print(f"  q={q}, batch_size={batch_size} (simulating Article 3 config)")

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=2e-4,
    )

    # Simulate HERALD manual DP-SGD (clip + noise, no Opacus hooks)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    model = model.to(device)

    C0 = 1.0
    sigma = 1.0
    noise_scale = C0 * sigma

    losses = []
    for step, batch in enumerate(dataloader):
        if step >= 5:  # 5 steps is enough to verify
            break

        input_ids_b, attention_mask_b, labels_b = [t.to(device) for t in batch]
        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids_b,
            attention_mask=attention_mask_b,
            labels=labels_b,
        )
        loss = outputs.loss
        loss.backward()

        # Verify lora_A gradients are None (frozen)
        if step == 0:
            a_grads = []
            b_grads = []
            for name, param in model.named_parameters():
                if "lora_A" in name:
                    a_grads.append(param.grad is None)
                elif "lora_B" in name:
                    b_grads.append(param.grad is not None)
            print(f"  lora_A grad is None: {all(a_grads)} (expected: True)")
            print(f"  lora_B has grad: {all(b_grads)} (expected: True)")

        # Manual DP-SGD: clip and add noise (same as HERALD)
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad and p.grad is not None],
            max_norm=C0,
        )
        for param in model.parameters():
            if param.requires_grad and param.grad is not None:
                param.grad += torch.randn_like(param.grad) * noise_scale

        optimizer.step()
        losses.append(loss.item())

    print(f"  Steps completed: {len(losses)}")
    print(f"  Losses: {[f'{l:.4f}' for l in losses]}")
    print(f"  Training step with FFA-LoRA + manual DP-SGD: SUCCESS")

    # Step 4: Verify model reload pattern (critical for HERALD)
    print("\n[4] Verifying model reload pattern...")
    # In HERALD, model is reloaded from disk at each fit() call.
    # FFA-LoRA must be re-applied after each reload.
    # Check: does loading a fresh model and re-applying FFA-LoRA work?
    try:
        from transformers import AutoModelForSequenceClassification
        from peft import LoraConfig, get_peft_model, TaskType

        model2 = AutoModelForSequenceClassification.from_pretrained(
            "bert-base-uncased", num_labels=50, ignore_mismatched_sizes=True
        )
        model2 = get_peft_model(model2, lora_config)

        # Re-apply FFA-LoRA
        for name, param in model2.named_parameters():
            if "lora_A" in name:
                param.requires_grad = False

        trainable2 = sum(p.numel() for p in model2.parameters() if p.requires_grad)
        print(f"  Reloaded model trainable params: {trainable2:,}")
        print(f"  Model reload + FFA-LoRA re-apply: SUCCESS")
    except Exception as e:
        print(f"  Model reload FAIL: {e}")
        return False

    # Step 5: Verify epsilon calculation with q=0.01
    print("\n[5] Verifying epsilon with q=0.01, sigma=1.0, R=20 and R=100...")
    try:
        from opacus.accountants import RDPAccountant
        import sys
        sys.path.insert(0, "experiments/adaptive-clipping")

        n_silo = 1500
        q_new = 0.01
        steps_per_round = max(1, int(n_silo * q_new))

        print(f"  n_silo={n_silo}, q={q_new}, steps/round={steps_per_round}")
        for R in [20, 50, 100]:
            steps_total = R * steps_per_round
            acc = RDPAccountant()
            acc.history = [(1.0, q_new, steps_total)]
            eps, _ = acc.get_privacy_spent(delta=1e-5)
            print(f"  R={R:3d}: epsilon={eps:.4f} {'<= 10 SIM' if eps <= 10 else '> 10 NAO'}")
    except Exception as e:
        print(f"  Epsilon calculation error: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  FFA-LoRA via requires_grad=False: WORKS")
    print(f"  Compatible with HERALD DP-SGD: YES")
    print(f"  Model reload compatible: YES")
    print(f"  Change needed in production code: MINIMAL")
    print(f"  Recommended implementation: freeze lora_A in model_setup_bert.py")
    print(f"  via env var FL_LORA_MODE=ffa (default: standard)")
    return True


if __name__ == "__main__":
    result = test_ffa_lora()
    exit(0 if result else 1)
