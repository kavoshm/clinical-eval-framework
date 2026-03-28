# Contributing to Clinical Eval Framework

Thank you for your interest in contributing. This project is a rubric-based LLM-as-judge evaluation harness for clinical AI outputs, with SQLite storage, version comparison, and regression detection. Contributions that improve rubric quality, scoring accuracy, or reporting are especially welcome.

## Development Setup

```bash
git clone https://github.com/kavosh-monfared/clinical-eval-framework.git
cd clinical-eval-framework
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

The demo pipeline runs in simulated mode (no API key required):

```bash
python src/main.py
```

To run with real LLM evaluation, set `OPENAI_API_KEY` in your environment or a `.env` file.

## Running Tests

```bash
# Full test suite
python -m pytest tests/ -v

# Specific modules
python -m pytest tests/test_models.py -v
python -m pytest tests/test_judge.py -v
python -m pytest tests/test_evaluator.py -v
python -m pytest tests/test_storage.py -v
```

All tests must pass before submitting a pull request.

## Code Style

- **Type hints** on all function signatures and return types.
- **Pydantic v2 models** for all data structures (`src/models.py`). Do not use raw dicts for scores, results, or reports.
- **Click CLI** conventions in `src/cli.py`. New commands should follow existing patterns.
- **YAML rubrics** in `rubrics/`. Follow the existing schema when adding or modifying rubrics.
- Keep evaluation logic deterministic where possible. Use `temperature=0` for LLM judge calls.

## Submitting Changes

1. Fork the repository and create a feature branch (`git checkout -b feature/your-feature`).
2. Make your changes with tests covering new behavior.
3. Run the full test suite and confirm all tests pass.
4. If you modified rubrics or scoring logic, run an evaluation and verify results are reasonable.
5. Open a pull request against `main` with a clear description of what changed and why.

## Clinical Safety Considerations

This framework evaluates clinical AI outputs where scoring errors can mask dangerous content. If your change modifies any of the following, take extra care:

- **Rubric definitions** (`rubrics/*.yaml`) -- Rubric weights, scoring levels, and criteria directly determine whether unsafe outputs are flagged. The patient safety rubric is weighted 2x for a reason. Do not lower safety weights or thresholds without strong justification.
- **Judge logic** (`src/judge.py`) -- Scoring heuristics in simulated mode and LLM prompts in real mode determine whether dangerous outputs (e.g., "labs are normal" when they are not) receive appropriately low scores.
- **Regression detection** (`src/evaluator.py`, `src/reporter.py`) -- The comparison engine must flag safety regressions with zero tolerance. Do not introduce changes that could suppress safety alerts.

A missed safety failure in the evaluation framework can propagate to downstream clinical systems. When in doubt, err on the side of stricter scoring.

## Questions

Open an issue with the `[QUESTION]` prefix if you have questions about the codebase or contribution process.
