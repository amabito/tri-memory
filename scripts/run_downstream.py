"""Downstream task evaluation via lm-evaluation-harness.

Evaluates the WT-103 pretrained TRNModel (44M, 10ep, PPL 55.79) on:
  lambada_openai, hellaswag, piqa, arc_easy, winogrande

Usage:
    python scripts/run_downstream.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import lm_eval
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model

from trimemory.block import CausalAttnBlock
from trimemory.config import TRNConfig
from trimemory.model import TRNModel


@register_model("trn")
class TRNModelWrapper(LM):
    """lm-eval wrapper for TRNModel."""

    def __init__(self, pretrained=None, device="cuda", batch_size=8, **kwargs):
        super().__init__()
        self._device = torch.device(device)
        self._batch_size = batch_size

        # Build model (same config as WT-103 10ep run)
        cfg = TRNConfig(
            vocab_size=50257, d_model=512, n_oscillators=256,
            n_layers=8, d_ff=1024, max_seq_len=256,
            dropout=0.0, gate_bias_init=0.65, state_norm=True, phase_mode="log",
        )
        torch.manual_seed(42)
        self.model = TRNModel(cfg).to(self._device)
        self.model.blocks[2] = CausalAttnBlock(cfg, n_heads=8).to(self._device)
        self.model.blocks[6] = CausalAttnBlock(cfg, n_heads=8).to(self._device)
        self.model.eval()

        # Load pretrained weights if available
        if pretrained and Path(pretrained).exists():
            state = torch.load(pretrained, map_location=self._device)
            self.model.load_state_dict(state)

        # Tokenizer
        from transformers import GPT2TokenizerFast
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return 256

    @property
    def max_gen_toks(self):
        return 64

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def device(self):
        return self._device

    def tok_encode(self, string, **kwargs):
        return self.tokenizer.encode(string)

    def tok_decode(self, tokens, **kwargs):
        return self.tokenizer.decode(tokens)

    def _model_call(self, inps):
        with torch.inference_mode():
            out = self.model(inps.to(self._device))
            return out["logits"]

    def _model_generate(self, context, max_length, stop, **kwargs):
        # Simple greedy generation
        ids = context.to(self._device)
        for _ in range(max_length - ids.shape[1]):
            with torch.inference_mode():
                logits = self.model(ids[:, -256:])["logits"][:, -1]
                next_tok = logits.argmax(dim=-1, keepdim=True)
                ids = torch.cat([ids, next_tok], dim=1)
        return ids

    def loglikelihood(self, requests, disable_tqdm=False):
        results = []
        for chunk_start in range(0, len(requests), self._batch_size):
            chunk = requests[chunk_start : chunk_start + self._batch_size]
            for req in chunk:
                context, continuation = req.args
                ctx_ids = self.tokenizer.encode(context)
                cont_ids = self.tokenizer.encode(continuation)
                full_ids = (ctx_ids + cont_ids)[-256:]  # truncate to max_length

                ids = torch.tensor([full_ids], device=self._device)
                with torch.inference_mode():
                    logits = self.model(ids)["logits"][0]  # (T, vocab)

                # Log-likelihood of continuation tokens
                cont_len = len(cont_ids)
                shift_logits = logits[-(cont_len + 1) : -1]  # (cont_len, vocab)
                shift_labels = torch.tensor(cont_ids, device=self._device)
                log_probs = F.log_softmax(shift_logits, dim=-1)
                token_lls = log_probs.gather(1, shift_labels.unsqueeze(1)).squeeze(1)
                ll = token_lls.sum().item()
                is_greedy = (shift_logits.argmax(dim=-1) == shift_labels).all().item()
                results.append((ll, is_greedy))
        return results

    def loglikelihood_rolling(self, requests, disable_tqdm=False):
        results = []
        for req in requests:
            (text,) = req.args
            ids = self.tokenizer.encode(text)[-256:]
            ids_t = torch.tensor([ids], device=self._device)
            with torch.inference_mode():
                logits = self.model(ids_t)["logits"][0]
            shift_logits = logits[:-1]
            shift_labels = torch.tensor(ids[1:], device=self._device)
            log_probs = F.log_softmax(shift_logits, dim=-1)
            ll = log_probs.gather(1, shift_labels.unsqueeze(1)).squeeze(1).sum().item()
            results.append((ll,))
        return results

    def generate_until(self, requests, disable_tqdm=False):
        results = []
        for req in requests:
            context = req.args[0]
            until = req.args[1] if len(req.args) > 1 else {"until": ["\n"]}
            ctx_ids = self.tokenizer.encode(context)[-200:]
            ids = torch.tensor([ctx_ids], device=self._device)
            for _ in range(self.max_gen_toks):
                with torch.inference_mode():
                    logits = self.model(ids[:, -256:])["logits"][:, -1]
                next_tok = logits.argmax(dim=-1, keepdim=True)
                ids = torch.cat([ids, next_tok], dim=1)
                decoded = self.tokenizer.decode(ids[0, len(ctx_ids):].tolist())
                if any(s in decoded for s in until.get("until", ["\n"])):
                    break
            results.append(decoded)
        return results


def main():
    # Note: model is randomly initialized (no pretrained checkpoint saved)
    # This measures the architecture's capability without WT-103 pretraining
    print("Evaluating TRN model (random init) on downstream tasks...")
    print("NOTE: No pretrained weights loaded -- results reflect random init baseline.")
    print()

    results = lm_eval.simple_evaluate(
        model="trn",
        tasks=["lambada_openai", "hellaswag", "piqa", "arc_easy", "winogrande"],
        batch_size=4,
        device="cuda",
        num_fewshot=0,
    )

    print("\n=== Results ===")
    for task_name, task_result in results["results"].items():
        acc = task_result.get("acc,none", task_result.get("acc_norm,none", "N/A"))
        print(f"  {task_name}: {acc}")

    import json
    with open("data/downstream_results.json", "w") as f:
        json.dump({k: {kk: vv for kk, vv in v.items() if isinstance(vv, (int, float, str))}
                   for k, v in results["results"].items()}, f, indent=2)
    print("\nSaved to data/downstream_results.json")


if __name__ == "__main__":
    main()
