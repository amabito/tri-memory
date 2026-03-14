from __future__ import annotations

from trimemory.config import TRNConfig


def test_d_ff_hidden_is_two_thirds_scaled():
    # d_ff=3072 -> raw=int(2/3*3072)=2048, already multiple of 256 -> 2048
    cfg = TRNConfig(d_ff=3072)
    assert cfg.d_ff_hidden == 2048


def test_d_ff_hidden_always_multiple_of_256():
    for preset in [TRNConfig.toy, TRNConfig.trn_100m, TRNConfig.trn_400m, TRNConfig.trn_1b]:
        cfg = preset()
        assert cfg.d_ff_hidden % 256 == 0, (
            f"{preset.__name__}: d_ff_hidden={cfg.d_ff_hidden} not multiple of 256"
        )


def test_d_ff_hidden_always_positive():
    for preset in [TRNConfig.toy, TRNConfig.trn_100m, TRNConfig.trn_400m, TRNConfig.trn_1b]:
        cfg = preset()
        assert cfg.d_ff_hidden > 0, f"{preset.__name__}: d_ff_hidden={cfg.d_ff_hidden} not positive"


def test_d_ff_hidden_less_than_or_equal_d_ff():
    # d_ff_hidden is 2/3 of d_ff rounded up to multiple of 256.
    # For small d_ff values the rounding can make d_ff_hidden == d_ff (e.g. toy: 512->512).
    # It should never exceed d_ff.
    for preset in [TRNConfig.toy, TRNConfig.trn_100m, TRNConfig.trn_400m, TRNConfig.trn_1b]:
        cfg = preset()
        assert cfg.d_ff_hidden <= cfg.d_ff, (
            f"{preset.__name__}: d_ff_hidden={cfg.d_ff_hidden} > d_ff={cfg.d_ff}"
        )


def test_all_presets_construct():
    # Should not raise
    TRNConfig.toy()
    TRNConfig.trn_100m()
    TRNConfig.trn_400m()
    TRNConfig.trn_1b()


def test_toy_config_fields():
    cfg = TRNConfig.toy()
    assert cfg.d_model == 128
    assert cfg.n_layers == 2
    assert cfg.vocab_size == 256
    assert cfg.n_oscillators == 64
    assert cfg.d_ff == 512
    assert cfg.max_seq_len == 512


def test_d_ff_hidden_non_standard():
    # d_ff=1000 -> raw=int(2/3*1000)=666, rounded up to next multiple of 256 -> 768
    cfg = TRNConfig(d_ff=1000)
    raw = int(2 / 3 * 1000)  # 666
    expected = (raw + 255) // 256 * 256  # 768
    assert cfg.d_ff_hidden == expected
