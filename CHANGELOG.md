# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
uses [Semantic Versioning](https://semver.org/).

## [2.0.1] - 2026-09-05

### Fixed

- The blink detector could lock into a permanent "eyes closed" state for
  people whose open-eye aspect ratio sits below the default threshold,
  because its baseline only learned from frames already judged open. The
  baseline is now a high percentile of all recent samples.
- Heart rate, emotion and affect were carried indefinitely after the face
  left the frame and recorded against faceless frames; they now expire after
  a short hold, and report statistics ignore frames without a face.
- PERCLOS, blink rate and time-in-state attributed a whole face-loss gap to
  the state of a single frame; gaps are now excluded from every time-based
  statistic, and long gaps restart the affect smoothers.
- `affectlab image` crashed on an assertion when the emotion backend could
  not produce an estimate (for example `--backend facs` without landmarks).
- Annotated video output silently dropped frames after the panel was
  toggled, never checked that the writer opened, and used the camera's
  nominal frame rate instead of the achieved one.
- A heart-rate window shorter than the minimum signal length could never
  produce an estimate; the minimum now follows the window, with a warning.
- Report figures no longer change the caller's matplotlib backend or style.
- The rPPG skin mask is rasterised over the face region instead of the
  whole frame.

## [2.0.0] - 2026-09-05

A rewrite from a single-script webcam demo into `affectlab`, an installable,
tested affective-computing toolkit. The scientific framing changed as much as
the code: facial movement is described (Action Units) before it is
interpreted (emotion labels), affect is tracked as continuous valence and
arousal, and contactless physiology sits alongside the face.

### Added

- `affectlab` package with a `src/` layout, `pyproject.toml`, typed public
  API and a console script (`affectlab live | video | image | report |
  doctor | models`).
- Facial Action Unit estimation (23 FACS AUs) from MediaPipe Face
  Landmarker blendshapes, with a documented mapping table.
- Three interchangeable emotion backends plus an ensemble: explainable
  EMFACS-style rules (`facs`), the FER+ convolutional network via ONNX
  Runtime (`ferplus`), and the original DeepFace pipeline (`deepface`,
  optional extra). Every estimate carries the evidence behind it.
- Dimensional affect on Russell's circumplex with time-aware smoothing, and
  emotion-dynamics statistics: inertia (lag-1 autocorrelation), variability,
  switch rate and time in state.
- Remote photoplethysmography heart rate with POS, CHROM and GREEN methods,
  spectral SNR quality grading, experimental RMSSD and breathing rate.
- Blink detection from the eye aspect ratio with an adaptive baseline, blink
  rate and PERCLOS.
- Head pose from MediaPipe's transformation matrix, with a PnP fallback in a
  matching frame convention.
- Live HUD with emotion bars, circumplex trajectory, AU bars, pulse waveform,
  vitals, dynamics, privacy blur and keyboard toggles; annotated video export.
- Fixed-schema CSV/JSONL session recording and a `report` command producing
  Markdown, JSON and four figures with a colour-vision-safe palette.
- Verified model downloads (SHA-256) into a local cache; YuNet fallback when
  MediaPipe is unavailable.
- Test suite (unit tests on synthetic ground truth, integration tests on a
  public-domain portrait and a synthetic pulse clip), GitHub Actions CI on
  Python 3.10 to 3.13, `ruff`, `mypy`, pre-commit configuration.
- Documentation: README, `docs/SCIENCE.md` (methods and references),
  `docs/ETHICS.md` (limits, law, consent), `docs/ARCHITECTURE.md`,
  `CONTRIBUTING.md`.

### Changed

- `main.py` is now a thin compatibility shim that launches `affectlab live`.
- DeepFace is optional (`pip install "affectlab[deepface]"`) instead of the
  only backend; the default is the FER+ and FACS ensemble.

### Removed

- The `tf_keras.py` shim. The DeepFace extra depends on `tf-keras` directly.

## [1.0.0] - 2025-02-19

- Initial release: real-time webcam emotion labels with DeepFace and OpenCV.
