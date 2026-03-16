"""seq_len throughput comparison: TRN O(n log n) vs Transformer O(n^2)."""
import sys, time, json
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from trimemory.config import TRNConfig
from trimemory.model import TRNModel
from trimemory.block import CausalAttnBlock

device = "cuda"

seq_lens = [128, 256, 512, 1024]

print(f"{'seq_len':>8} | {'TRN tok/s':>10} | {'TFM tok/s':>10} | {'Ratio':>8} | {'Winner':>10}")
print("-" * 60)

results = {}

for sl in seq_lens:
    bs = max(1, min(16, 32768 // sl))  # keep total tokens ~constant

    # TRN
    cfg = TRNConfig(vocab_size=50257, d_model=512, n_oscillators=256, n_layers=8,
                    d_ff=1024, max_seq_len=sl, dropout=0.0, gate_bias_init=0.65,
                    state_norm=True, phase_mode="log")
    torch.manual_seed(42)
    trn = TRNModel(cfg).to(device).eval()
    trn.blocks[2] = CausalAttnBlock(cfg, n_heads=8).to(device)
    trn.blocks[6] = CausalAttnBlock(cfg, n_heads=8).to(device)

    x = torch.randint(0, 50257, (bs, sl), device=device)
    # Warmup
    with torch.no_grad():
        for _ in range(3): trn(x)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    N = 20
    with torch.no_grad():
        for _ in range(N): trn(x)
        torch.cuda.synchronize()
    trn_tps = bs * sl * N / (time.perf_counter() - t0)
    del trn; torch.cuda.empty_cache()

    # Transformer
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_transformer_baseline import TransformerLM
    torch.manual_seed(42)
    tfm = TransformerLM(vocab_size=50257, d_model=512, n_heads=8, n_layers=8,
                        d_ff=1024, max_seq_len=sl, dropout=0.0).to(device).eval()
    x = torch.randint(0, 50257, (bs, sl), device=device)
    with torch.no_grad():
        for _ in range(3): tfm(x)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(N): tfm(x)
        torch.cuda.synchronize()
    tfm_tps = bs * sl * N / (time.perf_counter() - t0)
    del tfm; torch.cuda.empty_cache()

    ratio = tfm_tps / trn_tps
    winner = "TRN" if trn_tps > tfm_tps else "Transformer"
    results[sl] = {"trn": round(trn_tps), "tfm": round(tfm_tps), "ratio": round(ratio, 2)}
    print(f"{sl:8d} | {trn_tps:10.0f} | {tfm_tps:10.0f} | {ratio:7.2f}x | {winner}")

with open("data/seqlen_throughput.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to data/seqlen_throughput.json")
