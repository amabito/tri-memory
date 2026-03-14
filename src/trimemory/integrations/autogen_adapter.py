"""AutoGen adapter for TRN.

Provides TRNConversableAgent — a mixin that injects TRN long-term memory
into AutoGen conversable agents without requiring AutoGen at import time.

This module does NOT import autogen at load time. The actual framework
package is only referenced in type comments and docstrings.

Usage::

    # With AutoGen installed:
    #
    #   from autogen import ConversableAgent
    #   from trimemory.integrations import TRNConversableAgent
    #   from trimemory.config import TRNConfig
    #
    #   class MyAgent(TRNConversableAgent, ConversableAgent):
    #       pass
    #
    #   cfg = TRNConfig.trn_100m()
    #   agent = MyAgent(
    #       name="assistant",
    #       trn_cfg=cfg,
    #       trn_device="cpu",
    #       system_message="You are a helpful assistant.",
    #       llm_config={"model": "gpt-4"},
    #   )
    #
    #   # Feed conversation tokens manually (token IDs from your tokeniser):
    #   token_ids = tokeniser.encode(message)
    #   agent.trn_feed_tokens(token_ids)
    #
    #   # Retrieve current TRN state for inspection:
    #   state = agent.trn_get_state()
    #   print(f"State size: {agent.trn_state_size_bytes} bytes")
    #
    #   # Reset between sessions:
    #   agent.trn_reset()
"""
from __future__ import annotations

from typing import Any, Optional

import torch

from trimemory.config import TRNConfig
from trimemory.model import TRNModel


class TRNConversableAgent:
    """Mixin providing TRN long-term memory for AutoGen conversable agents.

    This mixin maintains an O(1) resonance state that is updated with each
    message token stream. It is framework-agnostic: any class that accepts
    keyword arguments in ``__init__`` can be combined with this mixin.

    The mixin intercepts ``__init__`` to initialise the TRN model and state.
    All TRN-specific kwargs (``trn_cfg``, ``trn_device``) are consumed here
    and not forwarded to the parent class.

    Memory characteristics:
        - State size: ``n_layers * n_oscillators * 2 * 4`` bytes (fp32).
        - For trn_100m: 8 layers * 256 oscillators * 2 * 4 = 16,384 bytes (16 KB).
        - KV cache equivalent at T=1000: ~8 MB (512x reduction).

    Args:
        trn_cfg: TRN architecture configuration.
        trn_device: Torch device for TRN state (default "cpu").
        **kwargs: Forwarded to the next class in MRO (e.g. ConversableAgent).

    Example::

        class TRNAgent(TRNConversableAgent, ConversableAgent):
            pass

        agent = TRNAgent(
            name="assistant",
            trn_cfg=TRNConfig.trn_100m(),
            trn_device="cpu",
            system_message="...",
            llm_config={...},
        )
        agent.trn_feed_tokens([42, 17, 99])
        state = agent.trn_get_state()
    """

    def __init__(
        self,
        *args: Any,
        trn_cfg: Optional[TRNConfig] = None,
        trn_device: str = "cpu",
        **kwargs: Any,
    ) -> None:
        # Initialise TRN components before calling parent __init__
        self._trn_cfg = trn_cfg or TRNConfig.trn_100m()
        self._trn_device = torch.device(trn_device)
        self._trn_model = TRNModel(self._trn_cfg).to(self._trn_device).eval()
        self._trn_position: int = 0
        self._trn_states_r = self._zero_states()
        self._trn_states_i = self._zero_states()

        # Forward remaining kwargs to the next class in MRO
        super().__init__(*args, **kwargs)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def trn_feed_tokens(self, token_ids: list[int]) -> None:
        """Feed a sequence of token IDs into the TRN resonance state.

        This is the primary method for updating agent memory. Call it
        after each message exchange to keep the state current.

        Args:
            token_ids: List of integer token IDs from your tokeniser.
        """
        if not token_ids:
            return

        tokens = torch.tensor(
            token_ids, dtype=torch.long, device=self._trn_device
        ).unsqueeze(0)  # (1, T)

        with torch.no_grad():
            for t in range(tokens.size(1)):
                token = tokens[:, t]
                x = self._trn_model.embedding(token)
                for layer_idx, block in enumerate(self._trn_model.blocks):
                    x_normed = block.norm1(x)
                    res_out, self._trn_states_r[layer_idx], self._trn_states_i[layer_idx] = (
                        block.resonance.step_single(
                            x_normed,
                            self._trn_states_r[layer_idx],
                            self._trn_states_i[layer_idx],
                            self._trn_position + t,
                        )
                    )
                    x = x + res_out
                    x = x + block.ffn(block.norm2(x))

        self._trn_position += len(token_ids)

    def trn_get_state(self) -> dict:
        """Return the current TRN state as serialisable nested lists.

        Returns:
            dict with keys ``r_real`` and ``r_imag``, each a list of
            ``n_layers`` tensors of shape ``(1, n_oscillators)`` as lists.
        """
        return {
            "r_real": [s.squeeze(0).tolist() for s in self._trn_states_r],
            "r_imag": [s.squeeze(0).tolist() for s in self._trn_states_i],
            "position": self._trn_position,
        }

    def trn_reset(self) -> None:
        """Reset TRN state and position counter (start a new session)."""
        self._trn_states_r = self._zero_states()
        self._trn_states_i = self._zero_states()
        self._trn_position = 0

    @property
    def trn_state_size_bytes(self) -> int:
        """Memory footprint of the TRN state in bytes."""
        return (
            self._trn_cfg.n_layers
            * self._trn_cfg.n_oscillators
            * 2  # real + imaginary
            * 4  # fp32
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _zero_states(self) -> list[torch.Tensor]:
        K = self._trn_cfg.n_oscillators
        return [
            torch.zeros(1, K, device=self._trn_device, dtype=torch.float32)
            for _ in range(self._trn_cfg.n_layers)
        ]
