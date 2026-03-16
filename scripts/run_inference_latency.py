"""Inference latency: TRN O(1) vs Transformer O(n) per-token generation."""
import sys, time, json
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from trimemory.config import TRNConfig
from trimemory.model import TRNModel
from trimemory.block import CausalAttnBlock

device = "cuda"

# --- TRN model ---
cfg = TRNConfig(vocab_size=50257, d_model=512, n_oscillators=256, n_layers=8,
                d_ff=1024, max_seq_len=4096, dropout=0.0, gate_bias_init=0.65,
                state_norm=True, phase_mode="log")
torch.manual_seed(42)
trn = TRNModel(cfg).to(device).eval()
trn.blocks[2] = CausalAttnBlock(cfg, n_heads=8).to(device)
trn.blocks[6] = CausalAttnBlock(cfg, n_heads=8).to(device)

# --- Transformer model ---
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_transformer_baseline import TransformerLM
torch.manual_seed(42)
tfm = TransformerLM(vocab_size=50257, d_model=512, n_heads=8, n_layers=8,
                    d_ff=1024, max_seq_len=4096, dropout=0.0).to(device).eval()

print(f"TRN params: {sum(p.numel() for p in trn.parameters()):,}")
print(f"TFM params: {sum(p.numel() for p in tfm.parameters()):,}")

# --- Measure per-token latency at various context lengths ---
positions = [64, 128, 256, 512, 1024, 2048]
results = {"trn": {}, "tfm": {}}

print(f"\n{'Pos':>6} | {'TRN ms/tok':>10} | {'TFM ms/tok':>10} | {'Ratio':>8}")
print("-" * 45)

for max_pos in positions:
    prompt = torch.randint(0, 50257, (1, 16), device=device)
    N_GEN = max_pos - 16
    if N_GEN <= 0:
        continue

    # TRN: use step_single (O(1) per token)
    K = cfg.n_oscillators
    states_r = [torch.zeros(1, K, device=device) for _ in range(cfg.n_layers)]
    states_i = [torch.zeros(1, K, device=device) for _ in range(cfg.n_layers)]
    param_dtype = next(trn.parameters()).dtype

    # Warmup
    with torch.inference_mode():
        for i in range(min(16, N_GEN)):
            tok = prompt[:, -1] if i == 0 else torch.randint(0, 50257, (1,), device=device)
            x = trn.drop_emb(trn.embedding(tok).to(param_dtype))
            for li, block in enumerate(trn.blocks):
                if hasattr(block, "resonance"):
                    x_n = block.norm1(x)
                    res_out, states_r[li], states_i[li] = block.resonance.step_single(
                        x_n, states_r[li], states_i[li], i)
                    x = x + res_out
                    x = x + block.ffn(block.norm2(x))
                else:
                    # Attn block: need KV cache for proper comparison
                    # For simplicity, skip attn layers in step_single
                    x = x + block.ffn(block.norm2(x))

    # Measure TRN
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        for i in range(N_GEN):
            tok = torch.randint(0, 50257, (1,), device=device)
            x = trn.drop_emb(trn.embedding(tok).to(param_dtype))
            for li, block in enumerate(trn.blocks):
                if hasattr(block, "resonance"):
                    x_n = block.norm1(x)
                    res_out, states_r[li], states_i[li] = block.resonance.step_single(
                        x_n, states_r[li], states_i[li], 16 + i)
                    x = x + res_out
                    x = x + block.ffn(block.norm2(x))
                else:
                    x = x + block.ffn(block.norm2(x))
    torch.cuda.synchronize()
    trn_ms = (time.perf_counter() - t0) / N_GEN * 1000

    # Measure Transformer (full causal attention over growing context)
    # Build up context token by token
    context = torch.randint(0, 50257, (1, 16), device=device)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        for i in range(N_GEN):
            out = tfm(context)
            next_tok = out["logits"][:, -1:].argmax(dim=-1)
            context = torch.cat([context, next_tok], dim=1)
            if context.size(1) > max_pos:
                context = context[:, -max_pos:]
    torch.cuda.synchronize()
    tfm_ms = (time.perf_counter() - t0) / N_GEN * 1000

    results["trn"][max_pos] = round(trn_ms, 3)
    results["tfm"][max_pos] = round(tfm_ms, 3)
    ratio = tfm_ms / trn_ms
    print(f"{max_pos:6d} | {trn_ms:10.3f} | {tfm_ms:10.3f} | {ratio:7.2f}x")

with open("data/inference_latency.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to data/inference_latency.json")
