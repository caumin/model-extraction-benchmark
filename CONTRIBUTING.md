# Contributing

Thanks for contributing to Model Extraction Benchmark.

## Development Setup

```bash
pip install -e ".[dev]"
```

## Before Opening a PR

Run the minimum checks:

```bash
ruff check mebench/
ruff format --check mebench/
mypy mebench/
pytest tests/test_contract_validation.py -v
```

## Contract-Critical Rules

- Keep `1 query = 1 image` budget accounting invariant.
- Do not bypass output-mode validation (`soft_prob` vs `hard_top1`).
- Keep victim inference deterministic in `eval()` and `no_grad()` paths.
- Do not commit runtime artifacts (`runs/`, checkpoints, generated reports).

## Attack Implementation Notes

- Implement attack interfaces under `mebench/attackers/`.
- Add tests for new behavior and config validation.
- Follow existing code style and typing conventions.

## Pull Request Guidance

- Keep changes focused and minimal.
- Explain why the change is needed and which contract points are affected.
- Include relevant test evidence in the PR description.
