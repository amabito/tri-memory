# Contributing to TRN

Contributions are welcome. This document describes the process.

## Getting Started

```bash
git clone https://github.com/TODO/trn.git
cd trn
pip install -e ".[dev]"
pytest tests/ -x -q
```

All 252 tests must pass before submitting a PR.

## Code Style

- Python 3.10+, type hints on all function signatures
- Format with `ruff format .`, lint with `ruff check .`
- No magic numbers -- use named constants
- Keep files under 400 lines where practical

## Pull Request Process

1. Fork the repository and create a feature branch
2. Make your changes
3. Run the full test suite: `pytest tests/ -x -q`
4. Run linting: `ruff check . && ruff format --check .`
5. If you changed model behavior, run the smoke benchmark: `python scripts/bench_smoke.py`
6. Open a PR against `main` with a clear description of what changed and why

## Benchmark Expectations

If your change affects model architecture or training:

- Run `python scripts/bench_smoke.py` and confirm it passes
- Run `python scripts/bench_generate.py --gen-lens 256,512,1024` and report TPS numbers
- If training dynamics changed, run `python scripts/bench_train.py --tasks copy --steps 1000` and include the loss curve

Include benchmark results in your PR description.

## Architecture Changes

Changes to the core recurrence (`resonance.py`, `scan.py`, `oscillator.py`) require:

1. An issue or discussion describing the motivation and expected impact
2. Before/after benchmark results on at least one generalization task
3. Verification that the P0 stability tests still pass: `pytest tests/test_trn_stability.py -v`
4. No regression in generation speed or memory

Open an issue with the `research-discussion` label to propose changes before writing code.

## Adding Tests

- Place tests in `tests/test_<module>.py`
- Follow the naming convention `test_<target>_<condition>_<expected>()`
- Each test should exercise a distinct code branch
- Adversarial tests belong in `tests/test_adversarial.py` or `tests/test_generate_adversarial.py`

## Reporting Results

If you report experimental results:

- State the exact command used to produce them
- Include model size, step count, device, and seed
- Do not extrapolate small-scale results to large-scale claims
- Results on models < 10M parameters should be labeled as "small-scale validation"
