# Contributing

Thank you for considering a contribution. AffectLab is a research toolkit,
so the bar for a change is "measurably correct and honestly described"
rather than "looks impressive".

## Setting up

```bash
git clone https://github.com/Juuzoe/PythonEmotionDetecter.git
cd PythonEmotionDetecter
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
affectlab models download        # about 40 MB, verified by checksum
affectlab doctor
```

On Debian or Ubuntu, MediaPipe needs a few system libraries:
`sudo apt install libegl1 libgles2 libgl1 libglib2.0-0`.

## Everyday commands

| Command | What it does |
|---|---|
| `make test` | unit tests (no models, no camera) |
| `make test-all` | unit and integration tests |
| `make lint` | `ruff check` and `ruff format --check` |
| `make format` | fix formatting and auto-fixable lint |
| `make typecheck` | `mypy src` |
| `make demo` | annotate the sample portrait into `docs/images/` |

CI runs lint, mypy and the unit suite on Python 3.10 to 3.13. Please run
`make lint typecheck test` before opening a pull request.

## Ground rules

* **Every measurement cites its source.** A new metric, threshold or
  coordinate needs a reference in `docs/SCIENCE.md`. If you cannot cite it,
  label it experimental in the code, the HUD and the report.
* **Pure stages stay pure.** Nothing under `src/affectlab/` except
  `sources`, `hud`, `recorder`, `report`, `models` and `cli` may touch a
  camera, a window, a file or the network.
* **Tests use synthetic ground truth.** Feed a known signal in, assert the
  known answer comes out. Integration tests may use the public-domain
  portrait in `tests/data/`; do not add photographs of identifiable people.
* **No new models without a checksum.** Add a `ModelSpec` with URL, licence,
  size and SHA-256 to `models.py`.
* **Respect the ethics page.** Pull requests that add surveillance,
  identification, proctoring or hiring features will be declined. See
  `docs/ETHICS.md`.
* **Style.** `ruff` decides formatting. Type hints everywhere. Docstrings
  say *why*, not what the code obviously does. Plain language in user-facing
  text; no marketing adjectives.

## Reporting problems

Open an issue with your operating system, Python version, the output of
`affectlab doctor`, and the exact command that misbehaved. For wrong
readings, say what you expected and, if you can, attach a short recorded
session (`--record`), which contains numbers only.
