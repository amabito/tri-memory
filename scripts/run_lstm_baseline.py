"""LSTM baseline on WikiText-2 BPE -- same config as TRN for fair comparison."""
from __future__ import annotations
import json, math, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

def prepare_data():
    from datasets import load_dataset
    from transformers import GPT2TokenizerFast
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    def enc(split):
        ids = tokenizer.encode("\n".join(ds[split]["text"]))
        return torch.tensor(ids, dtype=torch.long)
    return enc("train"), enc("validation"), enc("test"), tokenizer.vocab_size

class LSTMModel(nn.Module):
    def __init__(self, vocab, d, n_layers, dropout):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.lstm = nn.LSTM(d, d, n_layers, batch_first=True, dropout=dropout if n_layers > 1 else 0)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.emb.weight  # tie
        nn.init.normal_(self.emb.weight, std=d**-0.5)
    def forward(self, x, labels=None):
        h = self.drop(self.emb(x))
        h, _ = self.lstm(h)
        h = self.drop(h)
        logits = self.head(h)
        result = {"logits": logits}
        if labels is not None:
            result["loss"] = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                labels[:, 1:].reshape(-1), ignore_index=-100)
        return result

@torch.inference_mode()
def evaluate(model, data, seq_len, bs, device):
    model.eval()
    total, n = 0.0, 0
    for s in range(0, len(data)-seq_len-1, seq_len*bs):
        batch = []
        for b in range(bs):
            off = s + b*seq_len
            if off+seq_len+1 > len(data): break
            batch.append(data[off:off+seq_len].unsqueeze(0))
        if not batch: break
        ids = torch.cat(batch).to(device)
        total += model(ids, labels=ids)["loss"].item()
        n += 1
    return math.exp(total / max(n,1))

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}, GPU: {torch.cuda.get_device_name() if device=='cuda' else 'N/A'}")
    train, val, test, vocab = prepare_data()
    print(f"Data: train={len(train):,}, val={len(val):,}, test={len(test):,}, vocab={vocab}")

    seq_len, bs = 256, 16
    model = LSTMModel(vocab, d=256, n_layers=4, dropout=0.1).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"LSTM: {n_params:,} params, d=256, L=4, dropout=0.1")

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)
    n_epochs = 20
    n_ex = (len(train)-1) // seq_len
    results = {"epochs": []}

    print(f"\n{'Epoch':>5} | {'Train Loss':>10} | {'Val PPL':>10} | {'Time':>6}")
    print("-" * 45)
    best_val = float("inf")
    t0 = time.perf_counter()
    for ep in range(n_epochs):
        # LR schedule
        if ep < 3: lr = 3e-4 * (ep+1)/3
        else:
            p = (ep-3)/max(1,n_epochs-3)
            lr = 3e-5 + 0.5*(3e-4-3e-5)*(1+math.cos(p*math.pi))
        for pg in optimizer.param_groups: pg["lr"] = lr

        model.train()
        indices = torch.randperm(n_ex)
        tloss, ns = 0.0, 0
        for i in range(0, len(indices)-bs, bs):
            batch = []
            for idx in indices[i:i+bs]:
                off = idx.item()*seq_len
                if off+seq_len+1 > len(train): continue
                batch.append(train[off:off+seq_len].unsqueeze(0))
            if len(batch) < bs: continue
            ids = torch.cat(batch).to(device)
            optimizer.zero_grad()
            loss = model(ids, labels=ids)["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tloss += loss.item(); ns += 1
        avg_loss = tloss/max(ns,1)
        vppl = evaluate(model, val, seq_len, bs, device)
        et = time.perf_counter() - t0
        m = " *" if vppl < best_val else ""
        best_val = min(best_val, vppl)
        results["epochs"].append({"epoch":ep,"train_loss":round(avg_loss,4),"val_ppl":round(vppl,2)})
        print(f"{ep:5d} | {avg_loss:10.4f} | {vppl:10.2f} | {et/60:5.1f}m{m}")

    tppl = evaluate(model, test, seq_len, bs, device)
    print(f"\nBest Val PPL: {best_val:.2f}")
    print(f"Test PPL:     {tppl:.2f}")
    print(f"Total: {(time.perf_counter()-t0)/60:.1f} min")
    results["final"] = {"best_val_ppl": round(best_val,2), "test_ppl": round(tppl,2)}
    Path("data").mkdir(exist_ok=True)
    with open("data/lstm_baseline.json","w") as f: json.dump(results,f,indent=2)

if __name__ == "__main__":
    main()
