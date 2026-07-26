"""
Supplementary diagnostic (not the delivered POC): isolates the ghost-mode
failure with a full traceback, and independently tests standard mode on a
fresh model instance (the original POC's fallback reused the same model
object that ghost mode had already partially hooked, confounding the result).
"""
import warnings
warnings.filterwarnings("ignore")

import torch
from transformers import AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from opacus import PrivacyEngine

model_name = "prajjwal1/bert-tiny"
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS, r=8, lora_alpha=16,
    target_modules=["query", "value"], lora_dropout=0.0, bias="none",
)

n_samples = 100
seq_len = 128
input_ids = torch.randint(0, 1000, (n_samples, seq_len))
attention_mask = torch.ones(n_samples, seq_len, dtype=torch.long)
labels = torch.randint(0, 50, (n_samples,))
dataset = TensorDataset(input_ids, attention_mask, labels)


def fresh_model():
    m = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=50, ignore_mismatched_sizes=True,
    )
    return get_peft_model(m, lora_config)


print("=" * 60)
print("Isolated test 1: GHOST mode, full traceback")
print("=" * 60)
model = fresh_model()
optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
privacy_engine = PrivacyEngine()
try:
    model_p, optimizer_p, dl_p = privacy_engine.make_private(
        module=model, optimizer=optimizer, data_loader=dataloader,
        noise_multiplier=1.0, max_grad_norm=1.0, grad_sample_mode="ghost",
    )
    print("GHOST mode: SUCCESS")
except Exception:
    import traceback
    traceback.print_exc()

print()
print("=" * 60)
print("Isolated test 2: STANDARD mode on a FRESH model")
print("=" * 60)
model2 = fresh_model()
optimizer2 = AdamW([p for p in model2.parameters() if p.requires_grad], lr=2e-4)
dataloader2 = DataLoader(dataset, batch_size=8, shuffle=True)
privacy_engine2 = PrivacyEngine()
try:
    model2_p, optimizer2_p, dl2_p = privacy_engine2.make_private(
        module=model2, optimizer=optimizer2, data_loader=dataloader2,
        noise_multiplier=1.0, max_grad_norm=1.0,
    )
    print("STANDARD mode attach: SUCCESS")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model2_p = model2_p.to(device)
    batch = next(iter(dl2_p))
    ii, am, lb = [t.to(device) for t in batch]
    optimizer2_p.zero_grad()
    out = model2_p(input_ids=ii, attention_mask=am, labels=lb)
    out.loss.backward()
    optimizer2_p.step()
    eps = privacy_engine2.get_epsilon(delta=1e-5)
    print(f"STANDARD mode training step: SUCCESS, loss={out.loss.item():.4f}, epsilon={eps:.6f}")
except Exception:
    import traceback
    traceback.print_exc()
