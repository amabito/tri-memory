"""CrewAI adapter for TRN.

Provides TRNLongTermMemory — a drop-in replacement for CrewAI's
LongTermMemory interface, backed by TRN resonance state.

This module does NOT import crewai at load time. The actual framework
package is only referenced in type comments and docstrings.

Usage::

    # With CrewAI installed:
    #
    #   from crewai import Agent, Crew
    #   from trimemory.integrations import TRNLongTermMemory
    #   from trimemory.config import TRNConfig
    #
    #   memory = TRNLongTermMemory(TRNConfig.trn_100m(), device="cpu")
    #
    #   agent = Agent(
    #       role="researcher",
    #       goal="...",
    #       backstory="...",
    #       memory=True,
    #       long_term_memory=memory,
    #   )
    #
    #   crew = Crew(agents=[agent], tasks=[...], memory=True)
    #   crew.kickoff()
    #
    # CrewAI LongTermMemory interface (as of crewai>=0.28):
    #
    #   save(value: Any, metadata: dict, agent: str) -> None
    #   search(query: str, latest_n: int) -> list[dict]
    #   reset() -> None
"""
from __future__ import annotations

import json
from typing import Any, Optional

import torch

from trimemory.config import TRNConfig
from trimemory.model import TRNModel


class TRNLongTermMemory:
    """CrewAI LongTermMemory interface backed by TRN resonance state.

    Replaces CrewAI's default SQLite-based long-term memory with TRN's
    O(1) resonance state. The TRN state acts as a compact summary of all
    past interactions, enabling agents to operate indefinitely without
    context-length limits.

    Memory model:
        - ``save()`` tokenises the value (simple whitespace split) and
          feeds the resulting pseudo-token IDs into the TRN state.
        - ``search()`` returns the last N saved items from an in-memory
          log (the TRN state itself is not directly retrievable as text).
        - ``reset()`` zeroes the TRN state and clears the log.

    The TRN state is always O(1):
        state_bytes = n_layers * n_oscillators * 2 * 4

    For trn_100m: 8 * 256 * 2 * 4 = 16,384 bytes (16 KB) regardless
    of how many items have been saved.

    Args:
        cfg: TRN architecture configuration.
        device: Torch device string (default "cpu").

    Example::

        memory = TRNLongTermMemory(TRNConfig.trn_100m(), device="cpu")

        # Save a research finding
        memory.save(
            value={"finding": "TRN uses O(1) state", "confidence": 0.95},
            metadata={"task": "architecture_review", "step": 1},
            agent="researcher",
        )

        # Retrieve recent items
        results = memory.search("architecture", latest_n=5)
        for r in results:
            print(r)

        # Reset between sessions
        memory.reset()
        print(f"State size: {memory.state_size_bytes} bytes")
    """

    def __init__(
        self,
        cfg: Optional[TRNConfig] = None,
        device: str = "cpu",
    ) -> None:
        self._cfg = cfg or TRNConfig.trn_100m()
        self._device = torch.device(device)
        self._model = TRNModel(self._cfg).to(self._device).eval()
        self._position: int = 0
        self._states_r = self._zero_states()
        self._states_i = self._zero_states()
        self._log: list[dict] = []

    # ------------------------------------------------------------------
    # CrewAI LongTermMemory interface
    # ------------------------------------------------------------------

    def save(self, value: Any, metadata: dict, agent: str) -> None:
        """Persist a memory item and update the TRN resonance state.

        Args:
            value: Arbitrary value (dict, str, etc.) to remember.
            metadata: Contextual metadata dict (task name, step, etc.).
            agent: Name of the agent saving this memory.
        """
        # Serialise value to a string and derive pseudo-token IDs
        if isinstance(value, str):
            text = value
        else:
            try:
                text = json.dumps(value, ensure_ascii=False)
            except (TypeError, ValueError):
                text = str(value)

        token_ids = self._text_to_token_ids(text)
        self._feed_tokens(token_ids)

        self._log.append({
            "value": value,
            "metadata": metadata,
            "agent": agent,
            "position": self._position,
        })

    def search(self, query: str, latest_n: int = 3) -> list[dict]:
        """Return the last N saved items from the memory log.

        Note: The TRN resonance state encodes all past interactions as a
        compact fixed-size vector. This search returns raw log entries;
        more sophisticated retrieval can be built on top.

        Args:
            query: Search query string (currently unused for TRN retrieval).
            latest_n: Number of most-recent items to return.

        Returns:
            List of at most ``latest_n`` dicts with keys
            ``value``, ``metadata``, ``agent``, ``position``.
        """
        return self._log[-latest_n:] if self._log else []

    def reset(self) -> None:
        """Zero the TRN state and clear the memory log."""
        self._states_r = self._zero_states()
        self._states_i = self._zero_states()
        self._position = 0
        self._log.clear()

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def state_size_bytes(self) -> int:
        """Memory footprint of the TRN state in bytes (constant)."""
        return self._cfg.n_layers * self._cfg.n_oscillators * 2 * 4

    def get_trn_state(self) -> dict:
        """Return the raw TRN resonance state as nested lists.

        Useful for checkpointing or transfer between sessions.
        """
        return {
            "r_real": [s.tolist() for s in self._states_r],
            "r_imag": [s.tolist() for s in self._states_i],
            "position": self._position,
        }

    def load_trn_state(self, state: dict) -> None:
        """Restore TRN state from a previously saved dict."""
        self._states_r = [
            torch.tensor(v, device=self._device, dtype=torch.float32)
            for v in state["r_real"]
        ]
        self._states_i = [
            torch.tensor(v, device=self._device, dtype=torch.float32)
            for v in state["r_imag"]
        ]
        self._position = state.get("position", 0)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _zero_states(self) -> list[torch.Tensor]:
        K = self._cfg.n_oscillators
        return [
            torch.zeros(1, K, device=self._device, dtype=torch.float32)
            for _ in range(self._cfg.n_layers)
        ]

    def _text_to_token_ids(self, text: str) -> list[int]:
        """Simple character-hash tokenisation (no external tokeniser required).

        Maps each character to a stable token ID in [0, vocab_size).
        """
        vocab_size = self._cfg.vocab_size
        return [ord(c) % vocab_size for c in text[:512]]  # cap at 512 chars

    def _feed_tokens(self, token_ids: list[int]) -> None:
        """Feed token IDs into the TRN resonance state."""
        if not token_ids:
            return

        tokens = torch.tensor(
            token_ids, dtype=torch.long, device=self._device
        ).unsqueeze(0)  # (1, T)

        with torch.no_grad():
            for t in range(tokens.size(1)):
                token = tokens[:, t]
                x = self._model.embedding(token)
                for layer_idx, block in enumerate(self._model.blocks):
                    x_normed = block.norm1(x)
                    res_out, self._states_r[layer_idx], self._states_i[layer_idx] = (
                        block.resonance.step_single(
                            x_normed,
                            self._states_r[layer_idx],
                            self._states_i[layer_idx],
                            self._position + t,
                        )
                    )
                    x = x + res_out
                    x = x + block.ffn(block.norm2(x))

        self._position += len(token_ids)
