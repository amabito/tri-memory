"""LangGraph adapter for TRN.

Provides TRNMemoryNode — a LangGraph-compatible node that wraps TRNModel
to maintain O(1) resonance state across graph edges.

This module does NOT import langgraph at load time. The actual framework
package is only referenced in type comments and docstrings.

Usage::

    # In your LangGraph graph definition:
    #
    #   from trn.integrations import TRNMemoryNode
    #   from trn.config import TRNConfig
    #   from langgraph.graph import StateGraph
    #
    #   cfg = TRNConfig.trn_100m()
    #   memory_node = TRNMemoryNode(cfg, device="cpu")
    #
    #   # Add as a node in the graph:
    #   builder = StateGraph(AgentState)
    #   builder.add_node("memory", memory_node)
    #   builder.add_edge("memory", "llm")
    #
    #   # State schema (TypedDict):
    #   class AgentState(TypedDict):
    #       messages: list[dict]
    #       token_ids: list[int]   # new tokens to feed
    #       trn_context: dict      # TRN state passed between nodes
    #
    #   # The node callable signature:
    #   #   __call__(self, state: dict) -> dict
    #   # Reads state["token_ids"], updates TRN, returns state["trn_context"].
"""
from __future__ import annotations

from typing import Any, Optional

import torch

from trn.config import TRNConfig
from trn.model import TRNModel


class TRNMemoryNode:
    """LangGraph-compatible node that maintains TRN resonance state.

    Designed to be used as a node callable in a LangGraph StateGraph.
    The node ingests new token IDs from the graph state, updates the
    TRN resonance state, and writes it back for downstream nodes.

    The TRN state is O(1) in memory regardless of conversation length,
    making this suitable for long-running agent graphs.

    Args:
        cfg: TRN architecture configuration.
        device: Torch device string (e.g. "cpu", "cuda").
        batch_size: Number of concurrent agent instances (default 1).

    State keys consumed (from incoming state dict):
        ``token_ids`` (list[int]): Token IDs to feed this step.
        ``trn_context`` (dict, optional): Saved TRN state from previous step.

    State keys produced (merged into outgoing state dict):
        ``trn_context`` (dict): Updated TRN state for downstream nodes.
            Contains ``r_real``, ``r_imag`` — per-layer resonance states
            as nested lists (n_layers x batch_size x n_oscillators).

    Example::

        # Single-agent graph
        cfg = TRNConfig.trn_100m()
        node = TRNMemoryNode(cfg, device="cpu")

        state = {"token_ids": [42, 17, 99]}
        state = node(state)
        # state["trn_context"] now contains resonance state

        # Next turn — state["trn_context"] is carried forward
        state["token_ids"] = [100, 200]
        state = node(state)
    """

    def __init__(
        self,
        cfg: TRNConfig,
        device: str = "cpu",
        batch_size: int = 1,
    ) -> None:
        self.cfg = cfg
        self.device = torch.device(device)
        self.batch_size = batch_size
        self._model = TRNModel(cfg).to(self.device).eval()
        self._position: int = 0

    # ------------------------------------------------------------------
    # Node callable (LangGraph interface)
    # ------------------------------------------------------------------

    def __call__(self, state: dict[str, Any]) -> dict[str, Any]:
        """Process incoming tokens and update TRN state.

        Reads ``state["token_ids"]`` and optional ``state["trn_context"]``.
        Returns a dict with updated ``trn_context``.
        """
        token_ids: list[int] = state.get("token_ids", [])
        trn_ctx: Optional[dict] = state.get("trn_context")

        if not token_ids:
            return {"trn_context": trn_ctx or self._empty_context()}

        # Restore or initialise resonance states
        states_r, states_i = self._load_context(trn_ctx)

        tokens = torch.tensor(token_ids, dtype=torch.long, device=self.device)
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0).expand(self.batch_size, -1)

        with torch.no_grad():
            for t in range(tokens.size(1)):
                token = tokens[:, t]
                x = self._model.embedding(token)
                for layer_idx, block in enumerate(self._model.blocks):
                    x_normed = block.norm1(x)
                    res_out, states_r[layer_idx], states_i[layer_idx] = (
                        block.resonance.step_single(
                            x_normed,
                            states_r[layer_idx],
                            states_i[layer_idx],
                            self._position + t,
                        )
                    )
                    x = x + res_out
                    x = x + block.ffn(block.norm2(x))

        self._position += tokens.size(1)

        new_ctx = self._save_context(states_r, states_i)
        return {"trn_context": new_ctx}

    # ------------------------------------------------------------------
    # State serialisation helpers
    # ------------------------------------------------------------------

    def _empty_context(self) -> dict:
        K = self.cfg.n_oscillators
        B = self.batch_size
        return {
            "r_real": [
                torch.zeros(B, K, device=self.device, dtype=torch.float32).tolist()
                for _ in range(self.cfg.n_layers)
            ],
            "r_imag": [
                torch.zeros(B, K, device=self.device, dtype=torch.float32).tolist()
                for _ in range(self.cfg.n_layers)
            ],
        }

    def _load_context(
        self, ctx: Optional[dict]
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        K = self.cfg.n_oscillators
        B = self.batch_size
        L = self.cfg.n_layers

        if ctx is None:
            return (
                [torch.zeros(B, K, device=self.device, dtype=torch.float32) for _ in range(L)],
                [torch.zeros(B, K, device=self.device, dtype=torch.float32) for _ in range(L)],
            )

        states_r = [
            torch.tensor(ctx["r_real"][i], device=self.device, dtype=torch.float32)
            for i in range(L)
        ]
        states_i = [
            torch.tensor(ctx["r_imag"][i], device=self.device, dtype=torch.float32)
            for i in range(L)
        ]
        return states_r, states_i

    def _save_context(
        self,
        states_r: list[torch.Tensor],
        states_i: list[torch.Tensor],
    ) -> dict:
        return {
            "r_real": [s.tolist() for s in states_r],
            "r_imag": [s.tolist() for s in states_i],
        }

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset position counter (start a new conversation)."""
        self._position = 0

    @property
    def state_size_bytes(self) -> int:
        """Memory footprint of one agent's TRN state in bytes."""
        return self.cfg.n_layers * self.cfg.n_oscillators * 2 * 4
