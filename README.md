# Automatic Lecture Video Chapter Indexing

This project is a Flask web application that automatically segments lecture videos into labelled, navigable chapters. It was developed as the primary artefact of a thesis investigating automated chapter indexing for recorded lectures. The codebase contains two complete segmentation pipelines, a comprehensive evaluation suite, and the LaTeX source for the thesis itself.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Two Pipelines: How We Got Here](#two-pipelines-how-we-got-here)
3. [Pipeline 1 — Text-First Tumbling Window (Baseline)](#pipeline-1--text-first-tumbling-window-baseline)
4. [Pipeline 2 — Visual Sliding-Window (Final / Main)](#pipeline-2--visual-sliding-window-final--main)
5. [Installation & Setup](#installation--setup)
6. [Quick Start](#quick-start)
7. [Project Structure](#project-structure)
8. [Configuration Reference](#configuration-reference)

---

## Project Overview

Lecture recordings are hard to navigate. Auto-Chaptering goes through a video and automatically produces a chapter index — a list of timestamped, labelled segments that can be clicked to jump directly to that point in the lecture. The process is fully automatic: no manual annotations, no slide deck required.

The system combines three complementary signals:

- **Visual change detection** — ResNet-18 frame embeddings catch slide transitions and topic changes that leave a visible trace.
- **Automatic transcription** — OpenAI Whisper turns spoken audio into timed word segments.
- **LLM labelling** — Qwen2.5-1.5B-Instruct reads each segment's transcript and writes a short chapter title and description in structured JSON.

---

## Two Pipelines: How We Got Here

The thesis research followed a classic iterative design process. A simple text-only baseline was implemented first, evaluated, and then replaced by a more principled visual-signal-driven approach. Both pipelines share the same Flask front-end and output format, so they are fully interchangeable.

### Why the baseline was built first

The text-first tumbling window (Pipeline 1, `app.ipynb` / `run_app.py`) is the simplest possible segmentation strategy: slice the transcript every 300 words and call each slice a chapter. It is easy to implement, deterministic, and fast. It also has an obvious weakness — word counts do not align with topic changes, so chapters straddle slide boundaries and the resulting segments often lack semantic coherence.

### Why the visual pipeline supersedes it

The visual sliding-window pipeline (Pipeline 2, `sw_visual_app.py`) addresses the core limitation by locating boundaries where the *visual content of the video actually changes*. A ResNet-18 encodes a frame every 15 seconds, a sliding window pools adjacent frame features, and cosine distance spikes between consecutive windows identify chapter boundaries. Because boundaries are now anchored to real content transitions, chapter duration becomes content-driven rather than word-count-driven, and between-chapter semantic similarity is measurably lower (meaning chapters are more distinct from one another).

`pipeline_test.py` and the `test_fig_*.py` scripts document this comparison quantitatively. The key finding: the visual pipeline produces chapters with ~30–40% higher duration variability and lower mean adjacent-chapter SBERT cosine similarity — both indicators that boundaries are tracking genuine topic changes rather than arbitrary word counts.

---

## Pipeline 1 — Text-First Tumbling Window (Baseline)

**Entry point:** `run_app.py` (runs `app.ipynb`) or `make run`

### How it works

1. **Audio extraction** — FFmpeg extracts a 16 kHz mono WAV from the uploaded MP4.
2. **Whisper transcription** — OpenAI Whisper (`base` model, CPU) produces timed word segments. Filler words are removed.
3. **Tumbling-window segmentation** — Whisper segments are accumulated in a buffer until the total word count reaches 300. Each full buffer becomes one chapter. The final (possibly short) buffer is kept as-is.
4. **SBERT embeddings + KeyBERT keywords** — `all-MiniLM-L6-v2` encodes each chapter; KeyBERT extracts the top-5 keywords.
5. **LLM chapter generation** — Qwen2.5-1.5B-Instruct writes a short title and description per chapter in JSON format.
6. **Key/skip classification** — A second LLM pass classifies each chapter as *key* (substantive technical content) or *skip* (intro, recap, admin, Q&A).

### Characteristics

| Property | Typical value |
|---|---|
| Segmentation signal | Word count only |
| Chapter count (60 min lecture) | ~12–18 |
| Mean chapter duration | ~3–5 min |
| Duration variability (std) | Low — chapters are roughly equal length |
| Between-chapter SBERT similarity | Higher — boundaries don't track topic changes |

### Limitations

Because boundaries are placed every 300 words regardless of topic, a single chapter can span multiple distinct topics, and adjacent chapters can cover the same topic from one sentence to the next. This motivates Pipeline 2.

---

## Pipeline 2 — Visual Sliding-Window (Final / Main)

**Entry point:** `sw_visual_app.py` or `make run-visual`

### How it works

The pipeline runs as a 6-step background job after video upload. Progress is streamed in real time to the browser UI.

#### Stage 1 — Audio Extraction
FFmpeg strips the audio track and writes a 16 kHz mono WAV file. Cached: the WAV is reused on subsequent runs of the same video.

#### Stage 2 — Whisper Transcription
OpenAI Whisper (`base` model, CPU) transcribes the audio into timed word segments. Filler words (*um*, *uh*, *like*, *you know*, *so*, *actually*, *basically*, *right*) are removed by regex. Segments are cached to JSON so re-runs skip this slow step.

#### Stage 3 — Visual Feature Extraction & Boundary Detection
A ResNet-18 (ImageNet weights, classifier head removed — `fc` replaced with `nn.Identity()`) encodes one frame every 15 seconds into a 512-dimensional feature vector. Frames are read with OpenCV and preprocessed with the standard ImageNet normalisation transform.

A sliding window of width 60 s and step 5 s (91.7% overlap) pools consecutive frame feature vectors. Cosine distance between adjacent window feature means is computed, Gaussian-smoothed (σ = 2), and `scipy.signal.find_peaks` locates prominence peaks (minimum prominence 0.01, minimum separation 60 s) that become chapter boundaries.

#### Stage 4 — Transcript Slicing & Keyword Extraction
Whisper segments are allocated to the chapters defined by the visual breakpoints. Chapters below 60 words are merged forward into the next chapter. SBERT (`all-MiniLM-L6-v2`) encodes the chapter text; KeyBERT extracts the top-5 keywords per chapter. The pre-LLM chapter features are saved to `Output/raw_chapters.json` for debugging.

#### Stage 5 — LLM Chapter Generation & Key/Skip Classification
Qwen2.5-1.5B-Instruct (CPU, float32) generates a short title (5–8 words) and a 1–2 sentence technical description for each chapter in JSON format, with up to 3 retries per chapter and a word-based fallback. A second LLM pass classifies each chapter as `"True"` (key) or `"False"` (skip) using a yes/no prompt.

#### Stage 6 — Save
The final chapter list is written to `Output/sw_visual/<stem>-chapters.json`. The UI reads this file to render the interactive chapter index alongside the embedded video player.

### Characteristics

| Property | Typical value |
|---|---|
| Segmentation signal | ResNet-18 visual cosine distance |
| Chapter count (60 min lecture) | ~8–14 |
| Mean chapter duration | ~4–8 min |
| Duration variability (std) | High — boundaries track content changes |
| Between-chapter SBERT similarity | Lower — chapters are semantically distinct |

### Key design parameters

| Parameter | Default | Description |
|---|---|---|
| `FRAME_SAMPLE_RATE` | 15 s | Seconds between sampled frames |
| `W_VIS` | 60 s | Sliding window width |
| `S_VIS` | 5 s | Sliding window step |
| `SMOOTH_SIGMA` | 2.0 | Gaussian smoothing σ for cosine-distance curve |
| `PEAK_PROMINENCE` | 0.01 | Minimum peak prominence for boundary detection |
| `PEAK_MIN_DIST_S` | 60 s | Minimum gap between chapter boundaries |
| `MIN_CHAPTER_WORDS` | 60 | Merge chapters shorter than this word count |
| `WHISPER_SIZE` | `base` | Whisper model size |
| `LLM_MODEL` | `Qwen/Qwen2.5-1.5B-Instruct` | LLM for title/description generation |

---

## Installation & Setup

### Prerequisites

- Python 3.11 (required — versions tested with `C:/Python311/python.exe`)
- FFmpeg on the system `PATH` (for audio extraction)
- 16 GB RAM recommended (the LLM runs on CPU)
- ~8 GB free disk space for model downloads (Whisper, SBERT, Qwen)

### Install FFmpeg

**Windows (Chocolatey):**
```
choco install ffmpeg -y
```

**macOS:**
```
brew install ffmpeg
```

**Linux:**
```
sudo apt install ffmpeg
```

### Python environment

**Option A — Makefile (Windows, recommended):**
```
make          # creates .venv, installs requirements.txt
make run      # starts the text-first baseline pipeline
make run-visual  # starts the visual sliding-window pipeline
```

**Option B — manual:**
```bash
python3.11 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**For figure generation only** (no Flask server needed):
```bash
pip install matplotlib seaborn
```

### Model downloads

All models are downloaded automatically from Hugging Face on first use:

| Model | Size | Downloaded by |
|---|---|---|
| `openai/whisper-base` | ~145 MB | `whisper.load_model("base")` |
| `sentence-transformers/all-MiniLM-L6-v2` | ~90 MB | `SentenceTransformer(...)` |
| `Qwen/Qwen2.5-1.5B-Instruct` | ~3.0 GB | `AutoModelForCausalLM.from_pretrained(...)` |

Set `HF_HOME` to a custom cache directory if disk space is limited.

---

## Quick Start

### Running the visual pipeline (recommended)
```bash
python sw_visual_app.py
# Open http://localhost:5000
```

Upload an `.mp4` via the web UI. The pipeline runs in the background; progress is shown in real time. Once complete, click any chapter card to jump to that timestamp in the embedded video player.

### Running the text-first baseline
```bash
python run_app.py      # extracts and executes app.ipynb
# Open http://localhost:5000
```

### Re-running on an already-uploaded video
In the UI, select a video from the sidebar and click "Re-process" — or POST to `/process/<video_name>`.

---

## Project Structure

```
Presentation.pptx             Thesis presentation Slides
sw_visual_app.py              Main Flask app — visual sliding-window pipeline (Pipeline 2)
run_app.py                    Launcher for the text-first baseline (Pipeline 1, runs app.ipynb)
app.ipynb                     Text-first tumbling-window pipeline as a Jupyter notebook
requirements.txt              Python dependencies
Makefile                      Windows setup and run shortcuts

static/
  videos/                     Uploaded MP4 files
  transcripts/                Whisper WAV files + segment JSON caches

templates/
  UI-template.html            Single-page web UI (video player + chapter sidebar)

Output/
  sw_visual/                  Chapter JSON files from Pipeline 2 (one per video)
  tumbling_window/            Chapter JSON files from Pipeline 1 (if run separately)
  raw_chapters.json           Pre-LLM chapter features (debug output from Pipeline 2)

```

---

## Figure & Test Scripts

Each `test_fig_*.py` script is self-contained: it runs unit tests on the relevant pipeline helpers, computes data points from real or synthetic transcript/chapter data, and then renders the corresponding thesis figure using matplotlib. Scripts can be run individually without the Flask server or any video files.

### Running a single figure script
```bash
python test_fig_sliding_window_segmentation.py
python test_fig_pipeline_comparison.py
# etc.
```

Output PNGs are written to `test_results_figures/` at 200 DPI (sized for A4 Word document).

### Additional packages needed for figure scripts
```bash
pip install matplotlib seaborn
```

### Figure script reference

| Script | Output | What it shows |
|---|---|---|
| `test_fig_baseline_pipeline.py` | `slide4_baseline_pipeline.png` | Text-first pipeline flow diagram |
| `test_fig_visual_pipeline.py` | `slide5_visual_pipeline.png` | Visual sliding-window pipeline flow diagram |
| `test_fig_sliding_window_segmentation.py` | `slide6_sliding_window_segmentation.png` | How the sliding window traverses a video and detects boundaries |
| `test_fig_pipeline_comparison.py` | `pipeline_comparison.png` | Head-to-head metrics: chapter count, mean duration, key%, for Weeks 01–06 |
| `test_fig_features_llm.py` | `slide8_features_llm.png` | ResNet-18 feature extraction and LLM stage detail |
| `test_fig_llm_prompt_output.py` | `slide8b_llm_prompt_output.png` | Example LLM prompt and structured JSON output |
| `test_fig_coherence_analysis.py` | `slide14_coherence_analysis.png` | Between-chapter SBERT cosine similarity analysis |
| `test_fig_key_results_v2.py` | `slide14_key_results_v2.png` | Key results summary across all evaluated lectures |
| `test_benchmark_w01_w06.py` | `benchmark_w01_w06.png` | Effect of window width/step parameters (W=01–06) |
| `test_llm_model_comparison.py` | `llm_model_comparison.png` | Qwen model ablation: load time, inference speed, JSON success rate |

---

## Evaluation Suite (`pipeline_test.py`)

`pipeline_test.py` is a comprehensive evaluation driver that re-runs both pipelines from raw transcripts and compares them against the existing saved chapter outputs.

### What it does

1. Parses all `*_transcript.txt` files in `static/transcripts/`.
2. Re-runs the text-first tumbling-window segmentation (300-word windows) from the transcripts.
3. Re-runs a visual-proxy segmentation — the same sliding-window cosine-peak algorithm as Pipeline 2, but using SBERT text embeddings instead of ResNet-18 frame features. This proxy runs in under 5 minutes without video files.
4. Loads existing chapter JSONs from `Output/` (the real Pipeline 2 outputs).
5. Computes metrics across all three: SBERT between-chapter similarity, duration statistics, key/skip ratio, and chapter count.
6. Saves `metrics_results.json` and generates 10 publication-quality PNG figures (R1–R10).

### Running the evaluation
```bash
python pipeline_test.py                # full evaluation + all 10 figures
python pipeline_test.py --no-figures   # metrics only (faster)
```

Requires transcripts in `static/transcripts/` and chapter JSONs in `Output/`. Run `sw_visual_app.py` on at least one video first to generate both.

### Generated figures (R1–R10)

| Figure | File | Description |
|---|---|---|
| R1 | `figR1_chapter_count.png` | Chapter count per video, all three approaches |
| R2 | `figR2_between_sbert.png` | Mean adjacent-chapter SBERT cosine similarity (lower = more distinct) |
| R3 | `figR3_duration_distributions.png` | Chapter duration histograms by pipeline and lecture |
| R4 | `figR4_cosine_distance_curves.png` | SBERT cosine-distance curves with detected breakpoints |
| R5 | `figR5_key_skip_breakdown.png` | Key vs skip chapter breakdown by pipeline |
| R6 | `figR6_duration_stats.png` | Mean and std chapter duration bars |
| R7 | `figR7_sbert_detail.png` | Per-boundary SBERT similarity for the highest-chapter-count video |
| R8 | `figR8_aggregate_summary.png` | Aggregate metrics across all lectures |
| R9 | `figR9_chapter_timelines.png` | Chapter timeline strips (text-first vs visual-proxy side by side) |
| R10 | `figR10_results_table.png` | Full numeric results table |

---

## LLM Model Ablation (`test_llm_model_comparison.py`)

`test_llm_model_comparison.py` benchmarks eight Qwen-family models (0.5 B–3 B parameters) against the chapter-labelling task from Pipeline 2, using Week 01 Systems Programming lecture content as input.

Metrics recorded per model: wall-clock load time, single-chapter inference time, estimated time for 10 chapters, peak RAM delta, JSON parse success rate, title word count, output token count, and tokens/second.

```bash
python test_llm_model_comparison.py              # unit tests + reference table (no downloads)
python test_llm_model_comparison.py --live       # download and benchmark all models (~2 h on CPU)
python test_llm_model_comparison.py --live --model M5   # benchmark one model
python test_llm_model_comparison.py --live --save-results results_qwen.json
```

`Qwen2.5-1.5B-Instruct` was selected for Pipeline 2 as the best balance of inference speed, JSON reliability, and title quality on a 16 GB CPU-only machine.

---

## Configuration Reference

All parameters that affect pipeline behaviour are collected at the top of `sw_visual_app.py`. Changing any of them does not require touching the pipeline logic:

| Parameter | Default | Notes |
|---|---|---|
| `FRAME_SAMPLE_RATE` | 15 s | Higher values speed up Stage 3 at the cost of temporal resolution |
| `W_VIS` | 60 s | Wider windows smooth over rapid visual changes |
| `S_VIS` | 5 s | Smaller step increases boundary precision; also increases compute |
| `SMOOTH_SIGMA` | 2.0 | Higher σ suppresses noise but can merge nearby boundaries |
| `PEAK_PROMINENCE` | 0.01 | Lower threshold detects more (potentially spurious) boundaries |
| `PEAK_MIN_DIST_S` | 60 s | Prevents two boundaries from being placed within one minute of each other |
| `MIN_CHAPTER_WORDS` | 60 | Short chapters (e.g., short Q&A inserts) are merged into the next one |
| `WHISPER_SIZE` | `base` | Options: `tiny`, `base`, `small`, `medium`, `large` |
| `LLM_MODEL` | `Qwen/Qwen2.5-1.5B-Instruct` | Any HF causal LM that supports chat templates |
| `SBERT_MODEL` | `all-MiniLM-L6-v2` | Used for KeyBERT keyword extraction |

---

## Thesis Generation

The LaTeX thesis source lives in `thesis_latex/thesis.tex`. Each chapter also has a standalone Python PDF generator (using ReportLab) for rapid iteration:

```bash
python thesis_ch1_introduction.py    # → Output/thesis_ch1_introduction.pdf
python thesis_ch3_methodology.py     # → Output/thesis_ch3_methodology.pdf
python generate_full_thesis.py       # → assembles all chapters into one PDF
python generate_results_pdf.py       # → standalone results section PDF
```

Thesis figures are generated by `thesis_figure_1.py` … `thesis_figure_6.py` and output as both PNG (embedded in the PDF) and standalone files in the root directory.

---

## API Routes

The Flask server exposes the following endpoints:

| Method | Route | Description |
|---|---|---|
| GET | `/` | Home page — video list |
| GET | `/video/<video_name>` | Chapter index view for an existing video |
| POST | `/upload` | Upload a new MP4 and start the pipeline; returns `{job_id, video_name}` |
| POST | `/process/<video_name>` | Re-process an already-uploaded video |
| GET | `/api/job/<job_id>` | Poll pipeline progress: `{status, step, progress, logs}` |

---

## Dependencies Summary

| Package | Version | Purpose |
|---|---|---|
| `flask` | ≥3.0 | Web framework |
| `openai-whisper` | latest | Speech-to-text transcription |
| `opencv-python` | 4.9.0.80 | Video frame extraction |
| `torch` | 2.6.0 | ResNet-18 inference (CPU) |
| `torchvision` | ≥0.17 | ResNet-18 model + transforms |
| `transformers` | 4.46.3 | Qwen LLM loading and inference |
| `sentence-transformers` | 2.7.0 | SBERT chapter embeddings |
| `keybert` | latest | Keyword extraction |
| `scipy` | ≥1.12 | Peak detection, Gaussian smoothing |
| `numpy` | ≥1.26.4, <2.0 | Numerical operations |
| `Pillow` | ≥10.0 | PIL image handling for ResNet preprocessing |
| `matplotlib` | any | Figure generation (test scripts only) |
| `ffmpeg-python` | any | FFmpeg Python bindings |

Full pinned versions are in `requirements.txt`.