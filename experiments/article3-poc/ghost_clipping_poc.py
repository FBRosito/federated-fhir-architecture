"""
POC: Ghost Clipping compatibility with PubMedBERT+LoRA in HERALD environment.
Tests whether Opacus ghost clipping mode works with LoRA-wrapped BERT models.
"""

import warnings

warnings.filterwarnings("ignore")


def test_ghost_clipping_with_lora() -> None:
    """Test Ghost Clipping with a tiny BERT+LoRA model (no download required)."""
    print("=" * 60)
    print("POC: Ghost Clipping + LoRA compatibility test")
    print("=" * 60)

    # Step 1: Load a tiny BERT model (bert-tiny is ~17MB, already cached or fast to download)
    print("\n[1] Loading tiny BERT model...")
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_name = "prajjwal1/bert-tiny"
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=50,
            ignore_mismatched_sizes=True,
        )
        AutoTokenizer.from_pretrained(model_name)
        print(
            f"  Model loaded: {sum(p.numel() for p in model.parameters()):,} total params"
        )
    except Exception as e:
        print(f"  FAIL: {e}")
        return False

    # Step 2: Apply LoRA (same config as HERALD)
    print("\n[2] Applying LoRA adapters...")
    try:
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=8,
            lora_alpha=16,
            target_modules=["query", "value"],
            lora_dropout=0.0,  # 0 for reproducibility in POC
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(
            f"  Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)"
        )
    except Exception as e:
        print(f"  FAIL: {e}")
        return False

    # Step 3: Create minimal dataloader
    print("\n[3] Creating minimal dataloader...")
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    n_samples = 100
    seq_len = 128
    input_ids = torch.randint(0, 1000, (n_samples, seq_len))
    attention_mask = torch.ones(n_samples, seq_len, dtype=torch.long)
    labels = torch.randint(0, 50, (n_samples,))
    dataset = TensorDataset(input_ids, attention_mask, labels)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    print(f"  Dataset: {n_samples} samples, batch_size=8")

    # Step 4: Setup optimizer
    print("\n[4] Setting up optimizer...")
    from torch.optim import AdamW

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=2e-4,
    )

    # Step 5: Attach PrivacyEngine with Ghost Clipping
    print("\n[5] Attaching Opacus PrivacyEngine with Ghost Clipping...")
    from opacus import PrivacyEngine

    privacy_engine = PrivacyEngine()

    # Try ghost mode first
    try:
        model_private, optimizer_private, dataloader_private = (
            privacy_engine.make_private(
                module=model,
                optimizer=optimizer,
                data_loader=dataloader,
                noise_multiplier=1.0,
                max_grad_norm=1.0,
                grad_sample_mode="ghost",  # Ghost Clipping
            )
        )
        print("  Ghost Clipping mode: SUCCESS")
        ghost_available = True
    except Exception as e:
        print(f"  Ghost Clipping mode: FAIL ({type(e).__name__}: {e})")
        print("  Trying standard mode as fallback...")
        ghost_available = False
        try:
            model_private, optimizer_private, dataloader_private = (
                privacy_engine.make_private(
                    module=model,
                    optimizer=optimizer,
                    data_loader=dataloader,
                    noise_multiplier=1.0,
                    max_grad_norm=1.0,
                )
            )
            print("  Standard mode: SUCCESS")
        except Exception as e2:
            print(f"  Standard mode also failed: {e2}")
            return False

    # Step 6: Run one forward+backward pass
    print("\n[6] Running one training step...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    model_private = model_private.to(device)

    try:
        batch = next(iter(dataloader_private))
        input_ids_b, attention_mask_b, labels_b = [t.to(device) for t in batch]

        optimizer_private.zero_grad()
        outputs = model_private(
            input_ids=input_ids_b,
            attention_mask=attention_mask_b,
            labels=labels_b,
        )
        loss = outputs.loss
        loss.backward()
        optimizer_private.step()

        epsilon = privacy_engine.get_epsilon(delta=1e-5)
        print(f"  Loss: {loss.item():.4f}")
        print(f"  Epsilon after 1 step: {epsilon:.6f}")
        print("  Step SUCCESS")
    except Exception as e:
        print(f"  Training step FAIL: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Step 7: Simulate model reload (critical HERALD pattern)
    print("\n[7] Testing model reload pattern (HERALD-specific)...")
    print("  In HERALD, model is reloaded at each fit() call.")
    print("  This means PrivacyEngine must be re-attached each round.")
    try:
        # Simulate what HERALD does: create new model, new optimizer, new engine
        model2 = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=50, ignore_mismatched_sizes=True
        )
        model2 = get_peft_model(model2, lora_config)
        optimizer2 = AdamW([p for p in model2.parameters() if p.requires_grad], lr=2e-4)
        privacy_engine2 = PrivacyEngine()
        mode = "ghost" if ghost_available else None
        kwargs = {"grad_sample_mode": mode} if mode else {}
        model2_private, optimizer2_private, dataloader2_private = (
            privacy_engine2.make_private(
                module=model2,
                optimizer=optimizer2,
                data_loader=DataLoader(dataset, batch_size=8, shuffle=True),
                noise_multiplier=1.0,
                max_grad_norm=1.0,
                **kwargs,
            )
        )
        model2_private = model2_private.to(device)
        batch2 = next(iter(dataloader2_private))
        input_ids_b2, attention_mask_b2, labels_b2 = [t.to(device) for t in batch2]
        optimizer2_private.zero_grad()
        outputs2 = model2_private(
            input_ids=input_ids_b2,
            attention_mask=attention_mask_b2,
            labels=labels_b2,
        )
        outputs2.loss.backward()
        optimizer2_private.step()
        print("  Model reload + re-attach: SUCCESS")
        print("  (PrivacyEngine re-created each round is safe)")
    except Exception as e:
        print(f"  Model reload FAIL: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(
        f"  Ghost Clipping available: {'YES' if ghost_available else 'NO (fell back to standard)'}"
    )
    print("  LoRA + Opacus: COMPATIBLE")
    print("  Model reload pattern: COMPATIBLE")
    print(
        f"  Ready for Artigo 3 experiments: {'YES' if ghost_available else 'PARTIALLY (without ghost)'}"
    )
    return ghost_available


if __name__ == "__main__":
    result = test_ghost_clipping_with_lora()
    exit(0 if result else 1)
