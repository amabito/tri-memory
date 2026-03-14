from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TRNConfig:
    vocab_size:        int   = 32_000
    d_model:           int   = 768
    n_oscillators:     int   = 256       # K: resonance channels per layer
    n_layers:          int   = 12
    d_ff:              int   = 3_072     # raw FFN expansion target (pre-SwiGLU scaling)
    max_seq_len:       int   = 2_048
    dropout:           float = 0.0
    use_parallel_scan: bool  = True
    tie_weights:       bool  = True

    # --- Stabilization flags ---
    clamp_resonance:     bool  = False   # legacy L2 norm clamp (superseded by state_norm)
    resonance_clamp_val: float = 10.0    # max L2 norm of resonance state per oscillator

    # --- P0 stabilization (gradient explosion fix) ---
    amplitude_max:       float = 3.0     # softplus clamp ceiling (was 10.0)
    state_norm:          bool  = True    # per-channel max-abs normalization after state update
    res_scale_init:      float = 0.05    # learnable resonance output scale initial value
    gate_bias_init:      float = 0.65    # sigmoid(0.619) ~ 0.65; P0 stabilization default
    phase_mode:          str   = "log"   # "log" = omega*log(i+1), "linear" = omega*i
    scan_chunk_size:     int   = 16     # chunk size for chunked_resonance_scan

    @property
    def d_ff_hidden(self) -> int:
        """Actual hidden dim for SwiGLU gate/up projections.

        LLaMA-style: hidden = (2/3) * d_ff, rounded to next multiple of 256.
        This ensures total parameter count matches a conventional FFN whose
        expansion ratio is d_ff / d_model.
        """
        raw = int(2 / 3 * self.d_ff)
        return (raw + 255) // 256 * 256

    # --- Preset factories ---

    @classmethod
    def toy(cls) -> "TRNConfig":
        return cls(
            vocab_size=256, d_model=128, n_oscillators=64,
            n_layers=2, d_ff=512, max_seq_len=512,
        )

    @classmethod
    def trn_100m(cls) -> "TRNConfig":
        return cls(
            vocab_size=32_000, d_model=512, n_oscillators=256,
            n_layers=8, d_ff=2_048, max_seq_len=2_048,
        )

    @classmethod
    def trn_400m(cls) -> "TRNConfig":
        return cls(
            vocab_size=32_000, d_model=1_024, n_oscillators=512,
            n_layers=16, d_ff=4_096, max_seq_len=4_096,
        )

    @classmethod
    def trn_1b(cls) -> "TRNConfig":
        return cls(
            vocab_size=32_000, d_model=2_048, n_oscillators=512,
            n_layers=24, d_ff=8_192, max_seq_len=4_096,
        )
