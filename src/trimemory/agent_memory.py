"""AgentMemory: TRNModel wrapper for agent/chatbot use.

Manages resonance states internally so callers handle only token IDs,
not raw state tensors. Designed for streaming single-token inference.
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

import torch
from torch import Tensor

from .config import TRNConfig
from .model import TRNModel


class AgentMemory:
    """Wraps TRNModel for incremental agent inference.

    States are maintained across add_token() calls, enabling stateful
    conversation without re-processing the full history each step.

    Example::
        mem = AgentMemory(TRNConfig.toy(), device="cpu")
        for tok in prompt_ids:
            mem.add_token(tok)
        state = mem.get_state()
        mem.save("checkpoint.pt")
    """

    def __init__(
        self,
        cfg: TRNConfig,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        self.cfg = cfg
        self.device = torch.device(device)

        self.model = TRNModel(cfg).to(self.device)
        self.model.eval()

        K = cfg.n_oscillators
        self._states_r: list[Tensor] = [
            torch.zeros(1, K, device=self.device, dtype=torch.float32)
            for _ in range(cfg.n_layers)
        ]
        self._states_i: list[Tensor] = [
            torch.zeros(1, K, device=self.device, dtype=torch.float32)
            for _ in range(cfg.n_layers)
        ]
        self._position: int = 0

    # ------------------------------------------------------------------
    # Token ingestion
    # ------------------------------------------------------------------

    @torch.no_grad()
    def add_token(self, token_id: int) -> None:
        """Feed one token into the resonance state.

        Replicates the step_single loop from TRNModel.generate().
        """
        param_dtype = next(self.model.parameters()).dtype
        token = torch.tensor([token_id], device=self.device, dtype=torch.long)

        x = self.model.embedding(token).to(param_dtype)  # (1, d_model)
        x = self.model.drop_emb(x)

        for layer_idx, block in enumerate(self.model.blocks):
            x_normed = block.norm1(x)
            res_out, self._states_r[layer_idx], self._states_i[layer_idx] = (
                block.resonance.step_single(
                    x_normed,
                    self._states_r[layer_idx],
                    self._states_i[layer_idx],
                    self._position,
                )
            )
            x = x + res_out
            x = x + block.ffn(block.norm2(x))

        self._position += 1

    def add_tokens(self, token_ids: list[int]) -> None:
        """Feed multiple tokens sequentially."""
        for token_id in token_ids:
            self.add_token(token_id)

    # ------------------------------------------------------------------
    # State access
    # ------------------------------------------------------------------

    def get_state(self) -> dict:
        """Return current resonance states and position.

        Returns:
            dict with keys:
                - states_r: list of (1, K) fp32 tensors (one per layer)
                - states_i: list of (1, K) fp32 tensors (one per layer)
                - position: int, number of tokens processed
        """
        return {
            "states_r": [s.clone() for s in self._states_r],
            "states_i": [s.clone() for s in self._states_i],
            "position": self._position,
        }

    def reset(self) -> None:
        """Zero all resonance states and reset position counter."""
        for i in range(self.cfg.n_layers):
            self._states_r[i].zero_()
            self._states_i[i].zero_()
        self._position = 0

    def state_size_bytes(self) -> int:
        """Return memory used by resonance states in bytes.

        Formula: n_layers * K * 2 * 4  (two fp32 tensors, K elements each)
        """
        return self.cfg.n_layers * self.cfg.n_oscillators * 2 * 4

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Union[str, Path]) -> None:
        """Save resonance states and position to a file."""
        torch.save(
            {
                "states_r": [s.cpu() for s in self._states_r],
                "states_i": [s.cpu() for s in self._states_i],
                "position": self._position,
            },
            path,
        )

    def load(self, path: Union[str, Path]) -> None:
        """Load resonance states and position from a file saved by save()."""
        data = torch.load(path, map_location=self.device)
        self._states_r = [s.to(self.device) for s in data["states_r"]]
        self._states_i = [s.to(self.device) for s in data["states_i"]]
        self._position = int(data["position"])

    def to_dict(self) -> dict:
        """Serialize states to a plain Python dict (JSON-compatible).

        States are stored as nested lists of floats.
        """
        return {
            "states_r": [s.squeeze(0).tolist() for s in self._states_r],
            "states_i": [s.squeeze(0).tolist() for s in self._states_i],
            "position": self._position,
        }

    def from_dict(self, d: dict) -> None:
        """Restore states from a dict produced by to_dict()."""
        for i, vals in enumerate(d["states_r"]):
            self._states_r[i] = torch.tensor(
                vals, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
        for i, vals in enumerate(d["states_i"]):
            self._states_i[i] = torch.tensor(
                vals, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
        self._position = int(d["position"])
