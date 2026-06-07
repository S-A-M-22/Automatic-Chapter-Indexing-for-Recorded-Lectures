import json, os, uuid, threading, subprocess, traceback, re, time, warnings
from pathlib import Path
from flask import Flask, render_template, request, jsonify, url_for, redirect
import numpy as np

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_DATASETS_DISABLE_TORCHCODEC"] = "1"
os.environ["HF_DATASETS_AUDIO_BACKEND"] = "soundfile"

# ═══════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024   # 2 GB

UPLOAD_FOLDER     = os.path.join("static", "videos")
TRANSCRIPT_FOLDER = os.path.join("static", "transcripts")
OUTPUT_FOLDER     = "Output"
CHAPTER_DIR       = os.path.join(OUTPUT_FOLDER, "sw_visual")   # chapters live here
ALLOWED_EXT       = {".mp4"}

# -- Sliding-window visual segmentation parameters --
FRAME_SAMPLE_RATE = 15     # seconds between sampled frames
W_VIS             = 60     # sliding window width  (seconds)
S_VIS             = 5      # sliding window step   (seconds)
SMOOTH_SIGMA      = 2.0    # Gaussian sigma for smoothing the cosine-distance curve
PEAK_PROMINENCE   = 0.01   # minimum peak prominence for boundary detection
PEAK_MIN_DIST_S   = 60     # minimum seconds between consecutive chapter boundaries
MIN_CHAPTER_WORDS = 60     # merge chapters shorter than this word count forward

# -- Models --
WHISPER_SIZE = "base"
SBERT_MODEL  = "all-MiniLM-L6-v2"
LLM_MODEL    = "Qwen/Qwen2.5-1.5B-Instruct"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(TRANSCRIPT_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(CHAPTER_DIR, exist_ok=True)

# In-memory job tracker  {job_id: {status, step, progress, error, logs}}
jobs = {}

LOW_VALUE_KEYWORDS = [
    "introduction", "intro", "summary", "conclusion", "setup", "support",
    "student feedback", "welcome", "housekeeping", "announcements", "admin",
    "example", "installation", "environment setup", "recap", "review",
    "demo", "q&a", "off-topic", "discussion", "lab", "policies",
    "course overview", "tutorial", "basics", "overview", "tips",
    "preparation", "assignment", "project", "group formation",
    "lab allocation", "examples",
]


# ═══════════════════════════════════════════════════════════════
#  Helper utilities
# ═══════════════════════════════════════════════════════════════
def parse_time(t: str) -> int:
    parts = t.split(":")
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + int(float(s))
    if len(parts) == 2:
        m, s = parts
        return int(m) * 60 + int(float(s))
    return 0


def fmt_ts(seconds: float) -> str:
    s = max(0, int(seconds))
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def cosine_sim(a, b) -> float:
    """Cosine similarity between two 1-D numpy-compatible arrays."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if (na > 0 and nb > 0) else 0.0


def is_low_value(ch: dict) -> bool:
    text = (ch.get("chapter", "") + " " + ch.get("description", "")).lower()
    return any(k in text for k in LOW_VALUE_KEYWORDS)


def extract_json_brace(text: str):
    """Return the first complete {...} JSON object from *text*, or None."""
    count, start = 0, None
    for i, ch in enumerate(text):
        if ch == "{":
            if start is None:
                start = i
            count += 1
        elif ch == "}":
            count -= 1
            if count == 0 and start is not None:
                return text[start : i + 1]
    return None


def enrich_chapters(chapters: list) -> list:
    enriched = []
    for ch in chapters:
        s = parse_time(ch["start_time"])
        e = parse_time(ch["end_time"])
        key_str = ch.get("key", "")
        if key_str in ("True", "False"):
            low_value = (key_str == "False")
        else:
            low_value = is_low_value(ch)
        enriched.append({**ch, "start_sec": s, "end_sec": e, "low_value": low_value})
    return enriched


def find_chapter_file(stem: str, chapter_dir: str):
    """Find a chapter JSON file for the given video stem in chapter_dir.

    Tries the exact stem with common naming suffixes, then retries after
    stripping a trailing '-full' from the stem to match older files that
    were generated from truncated video names.
    """
    if not os.path.isdir(chapter_dir):
        return None

    # Candidate stems: original, then with trailing '-full' stripped
    stems = [stem]
    if stem.endswith("-full"):
        stems.append(stem[:-5])

    suffixes = [
        "-chapters.json",
        "-full-chapters.json",
        "-full-full-chapters.json",
    ]

    for s in stems:
        for suffix in suffixes:
            path = os.path.join(chapter_dir, s + suffix)
            if os.path.exists(path):
                return path

    return None


def get_available_videos() -> list:
    videos = []
    for f in Path(UPLOAD_FOLDER).iterdir():
        if f.suffix.lower() in ALLOWED_EXT:
            stem = f.stem
            has_ch = find_chapter_file(stem, CHAPTER_DIR) is not None
            videos.append({"name": f.name, "stem": stem, "has_chapters": has_ch})
    return sorted(videos, key=lambda v: v["name"])


def load_chapters(video_name: str) -> list:
    stem = Path(video_name).stem
    path = find_chapter_file(stem, CHAPTER_DIR)
    if path:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else data.get("chapters", [])
    return []


# ═══════════════════════════════════════════════════════════════
#  Pipeline helpers
# ═══════════════════════════════════════════════════════════════
def log_msg(job: dict, text: str) -> None:
    job["logs"].append({"t": time.strftime("%H:%M:%S"), "msg": text})
    job["step"] = text


_FILLER = re.compile(
    r"\b(?:um+|uh+|like|you know|so|actually|basically|right)\b", re.I
)

def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", _FILLER.sub("", text)).strip()


# ── LLM chapter generation ──────────────────────────────────────────────────
def generate_chapter_json(ch: dict, model, tokenizer, idx: int, max_retries: int = 3) -> dict:
    """Generate a chapter summary JSON for a visual-segment dict.

    Expected keys in *ch*: start_time, end_time, word_count, text, keywords.
    """
    import torch
    dur = parse_time(ch["end_time"]) - parse_time(ch["start_time"])
    prompt = (
        "Generate a chapter summary JSON for this lecture segment.\n\n"
        f"Time Range : {ch['start_time']} to {ch['end_time']}\n"
        f"Duration   : {dur} seconds\n"
        f"Word Count : {ch['word_count']}\n"
        f"Keywords   : {', '.join(ch.get('keywords', [])[:5]) or 'N/A'}\n\n"
        f"Transcript :\n{ch['text'][:600]}\n\n"
        "Requirements:\n"
        "- chapter     : 5-8 word descriptive title\n"
        "- description : 1-2 sentence technical summary (under 50 words)\n\n"
        "Output ONLY valid JSON. No markdown. No explanation."
    )
    messages = [
        {"role": "system", "content": "You are a precise JSON-generating assistant."},
        {"role": "user",   "content": prompt},
    ]
    raw_input = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(raw_input, return_tensors="pt")

    for attempt in range(max_retries):
        try:
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.1,
                    top_p=0.9,
                    do_sample=True,
                    repetition_penalty=1.1,
                    pad_token_id=tokenizer.eos_token_id,
                )
            generated = tokenizer.decode(
                out[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )
            json_str = extract_json_brace(generated)
            if json_str:
                json_str = re.sub(r",\s*}", "}", json_str)           # trailing commas
                json_str = re.sub(r":\s*True\b",  ': "True"',  json_str)
                json_str = re.sub(r":\s*False\b", ': "False"', json_str)
                obj = json.loads(json_str)
                if "chapter" in obj and "description" in obj:
                    obj["ID"]         = str(idx + 1)
                    obj["start_time"] = ch["start_time"]  # enforce correct timestamps
                    obj["end_time"]   = ch["end_time"]
                    return obj
        except (json.JSONDecodeError, Exception):
            pass

    # Fallback
    return {
        "ID":          str(idx + 1),
        "chapter":     " ".join(ch["text"].split()[:7]),
        "start_time":  ch["start_time"],
        "end_time":    ch["end_time"],
        "description": "LLM parse failed -- fallback title.",
    }


def classify_high_level(chapter: dict, model, tokenizer) -> str:
    import torch

    title_desc = (chapter.get("chapter", "") + " " + chapter.get("description", "")).lower()
    if any(k in title_desc for k in LOW_VALUE_KEYWORDS):
        return "False"

    prompt = (
        "Is this lecture chapter a key technical topic (not welcome/recap/tutorial/admin/Q&A)?\n"
        f"Title: {chapter.get('chapter')}\n"
        f"Description: {chapter.get('description')}\n"
        "Answer only Yes or No."
    )
    try:
        inp = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            out = model.generate(
                **inp,
                max_new_tokens=8,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        raw = tokenizer.decode(out[0], skip_special_tokens=True)
        if re.search(r"\b(yes|true)\b", raw, re.IGNORECASE):
            return "True"
    except Exception:
        pass
    return "False"


# ═══════════════════════════════════════════════════════════════
#  Background pipeline runner  (6 steps)
# ═══════════════════════════════════════════════════════════════
def run_pipeline(job_id: str, video_path: str, video_name: str):
    job  = jobs[job_id]
    stem = Path(video_name).stem
    audio_path      = os.path.join(TRANSCRIPT_FOLDER, f"{stem}.wav")
    transcript_path = os.path.join(TRANSCRIPT_FOLDER, f"{stem}_transcript.txt")
    segments_cache  = os.path.join(TRANSCRIPT_FOLDER, f"{stem}_segments.json")

    try:
        # ── 1  Audio extraction ─────────────────────────────────────────────
        job["progress"] = "1 / 6"
        if os.path.exists(audio_path):
            log_msg(job, f"[1/6] Audio already exists -> {os.path.basename(audio_path)} (skipped)")
        else:
            log_msg(job, "[1/6] Extracting audio via FFmpeg ...")
            res = subprocess.run(
                ["ffmpeg", "-y", "-i", video_path,
                 "-ar", "16000", "-ac", "1", "-f", "wav", audio_path],
                capture_output=True, text=True, timeout=600,
            )
            if res.returncode != 0:
                raise RuntimeError(f"FFmpeg error:\n{res.stderr[:500]}")
            log_msg(job, f"[1/6] Audio saved -> {os.path.basename(audio_path)}")

        # ── 2  Whisper transcription ───────────────────────────────────────
        job["progress"] = "2 / 6"
        if os.path.exists(segments_cache):
            log_msg(job, f"[2/6] Transcript cache found -> {os.path.basename(segments_cache)} (skipped)")
            with open(segments_cache, "r", encoding="utf-8") as f:
                segments = json.load(f)
        else:
            log_msg(job, f"[2/6] Loading Whisper {WHISPER_SIZE} ...")
            import whisper as _whisper
            wmodel = _whisper.load_model(WHISPER_SIZE)
            log_msg(job, "[2/6] Transcribing audio (this may take a while on CPU) ...")
            result = wmodel.transcribe(audio_path, language="en")
            del wmodel

            segments = []
            for seg in result.get("segments", []):
                t = clean_text(seg["text"].strip())
                if t:
                    segments.append({
                        "id":         seg["id"],
                        "start_time": float(seg["start"]),
                        "end_time":   float(seg["end"]),
                        "text":       t,
                    })
            if not segments:
                raise RuntimeError("Whisper produced no segments - is the audio silent?")

            # Save segment cache for future runs
            with open(segments_cache, "w", encoding="utf-8") as f:
                json.dump(segments, f, indent=2, ensure_ascii=False)
            # Human-readable transcript
            with open(transcript_path, "w", encoding="utf-8") as f:
                for seg in segments:
                    f.write(
                        f"[{fmt_ts(seg['start_time'])} --> {fmt_ts(seg['end_time'])}] "
                        f"{seg['text']}\n"
                    )

        video_end = float(segments[-1]["end_time"])
        log_msg(job, f"[2/6] Transcript: {len(segments)} segments, {fmt_ts(video_end)}")

        # ── 3  Visual features + sliding-window cosine peaks ────────────────
        job["progress"] = "3 / 6"
        log_msg(job, "[3/6] Loading ResNet-18 for visual feature extraction ...")
        import torch
        import cv2
        from torchvision import models, transforms
        from PIL import Image as PILImage
        from scipy.signal import find_peaks
        from scipy.ndimage import gaussian_filter1d

        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        resnet.fc = torch.nn.Identity()  # strip classifier -> 512-d feature vector
        resnet.eval()

        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
            ),
        ])

        log_msg(job, f"[3/6] Extracting frames every {FRAME_SAMPLE_RATE}s ...")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        fps      = cap.get(cv2.CAP_PROP_FPS) or 25.0
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step     = max(1, int(fps * FRAME_SAMPLE_RATE))
        ts_list, feat_list = [], []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % step == 0:
                ts_list.append(frame_idx / fps)
                rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                t    = preprocess(PILImage.fromarray(rgb)).unsqueeze(0)
                with torch.no_grad():
                    feat = resnet(t).squeeze().numpy()
                feat_list.append(feat)
                if len(feat_list) % 20 == 0:
                    log_msg(
                        job,
                        f"[3/6] Frames sampled: {len(feat_list)} "
                        f"({ts_list[-1]:.0f}s / ~{n_frames / fps:.0f}s)",
                    )
            frame_idx += 1
        cap.release()
        del resnet

        frame_ts   = np.array(ts_list, dtype=np.float32)
        frame_feat = np.stack(feat_list).astype(np.float32)
        log_msg(job, f"[3/6] {len(frame_ts)} frames -> shape {frame_feat.shape}")

        # Build sliding windows: width=W_VIS, step=S_VIS
        log_msg(
            job,
            f"[3/6] Building sliding windows  W={W_VIS}s  S={S_VIS}s "
            f"(overlap {100 * (W_VIS - S_VIS) / W_VIS:.0f}%) ...",
        )
        win_times, win_feats = [], []
        t_start = 0.0
        while t_start + W_VIS <= float(frame_ts[-1]):
            t_end = t_start + W_VIS
            mask  = (frame_ts >= t_start) & (frame_ts < t_end)
            if mask.sum() > 0:
                win_times.append(t_start + W_VIS / 2.0)     # window centre timestamp
                win_feats.append(frame_feat[mask].mean(axis=0))
            t_start += S_VIS

        win_times = np.array(win_times)
        win_feats = np.stack(win_feats)
        log_msg(job, f"[3/6] {len(win_times)} sliding windows built")

        # Cosine distance between consecutive windows
        cos_dist = np.array([
            1.0 - cosine_sim(win_feats[i], win_feats[i + 1])
            for i in range(len(win_feats) - 1)
        ], dtype=np.float64)
        trans_times = (win_times[:-1] + win_times[1:]) / 2.0

        # Gaussian smoothing + peak detection
        cos_smooth     = gaussian_filter1d(cos_dist, sigma=SMOOTH_SIGMA)
        min_dist_steps = max(1, int(PEAK_MIN_DIST_S / S_VIS))
        peak_idx, _    = find_peaks(
            cos_smooth, prominence=PEAK_PROMINENCE, distance=min_dist_steps,
        )
        breakpoints = trans_times[peak_idx]
        log_msg(
            job,
            f"[3/6] Detected {len(breakpoints)} visual breakpoints: "
            + ", ".join(fmt_ts(b) for b in breakpoints[:8])
            + (" ..." if len(breakpoints) > 8 else ""),
        )

        # ── 4  Transcript slicing + SBERT / KeyBERT ────────────────────────
        job["progress"] = "4 / 6"
        log_msg(job, "[4/6] Slicing transcript at visual breakpoints ...")

        bounds = np.concatenate([[0.0], np.sort(breakpoints), [float("inf")]])
        raw_chapters = []
        for i in range(len(bounds) - 1):
            t_lo, t_hi = float(bounds[i]), float(bounds[i + 1])
            segs = [
                s for s in segments
                if float(s["start_time"]) >= t_lo and float(s["start_time"]) < t_hi
            ]
            if not segs:
                continue
            text = " ".join(s["text"] for s in segs)
            raw_chapters.append({
                "start_sec":  float(segs[0]["start_time"]),
                "end_sec":    float(segs[-1]["end_time"]),
                "text":       text,
                "word_count": len(text.split()),
            })

        # Merge chapters below MIN_CHAPTER_WORDS forward
        merged = []
        i = 0
        while i < len(raw_chapters):
            ch = dict(raw_chapters[i])
            while ch["word_count"] < MIN_CHAPTER_WORDS and i + 1 < len(raw_chapters):
                i += 1
                nxt = raw_chapters[i]
                ch["text"]       += " " + nxt["text"]
                ch["word_count"] += nxt["word_count"]
                ch["end_sec"]     = nxt["end_sec"]
            merged.append(ch)
            i += 1

        for j, ch in enumerate(merged, start=1):
            ch["ID"]         = str(j)
            ch["start_time"] = fmt_ts(ch["start_sec"])
            ch["end_time"]   = fmt_ts(ch["end_sec"])

        log_msg(job, f"[4/6] {len(merged)} chapters after merge")

        # SBERT embeddings + KeyBERT keywords
        log_msg(job, f"[4/6] Loading SBERT ({SBERT_MODEL}) + KeyBERT ...")
        from sentence_transformers import SentenceTransformer
        from keybert import KeyBERT

        sbert    = SentenceTransformer(SBERT_MODEL)
        kw_model = KeyBERT(model=sbert)

        texts = [ch["text"][:1200] for ch in merged]
        sbert.encode(texts, batch_size=16, show_progress_bar=False, convert_to_numpy=True)
        log_msg(job, "[4/6] SBERT embeddings computed")

        log_msg(job, f"[4/6] Extracting keywords for {len(merged)} chapters ...")
        for i, ch in enumerate(merged):
            txt = clean_text(ch["text"])
            kws = [w for w, _ in kw_model.extract_keywords(txt, top_n=5, stop_words="english")] if txt else []
            ch["keywords"] = kws
            log_msg(job, f"[4/6] KeyBERT ch {i + 1}/{len(merged)}: {', '.join(kws[:3])}")

        del sbert, kw_model

        # Save raw chapter features before LLM
        raw_features_path = os.path.join(OUTPUT_FOLDER, "raw_chapters.json")
        raw_features = []
        for ch in merged:
            raw_features.append({
                "ID":           ch["ID"],
                "start_time":   ch["start_time"],
                "end_time":     ch["end_time"],
                "start_sec":    ch["start_sec"],
                "end_sec":      ch["end_sec"],
                "duration_sec": round(ch["end_sec"] - ch["start_sec"], 2),
                "word_count":   ch["word_count"],
                "keywords":     ch.get("keywords", []),
                "text":         ch["text"],
            })
        with open(raw_features_path, "w", encoding="utf-8") as f:
            json.dump(raw_features, f, indent=2, ensure_ascii=False)
        log_msg(job, f"[4/6] Raw chapter features saved -> {os.path.basename(raw_features_path)}")

        # ── 5  LLM chapter generation + key/skip classification ───────────
        job["progress"] = "5 / 6"
        log_msg(job, f"[5/6] Loading {LLM_MODEL} (CPU, float32) ...")
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL, trust_remote_code=True)
        llm       = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL,
            dtype=torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        llm.eval()
        log_msg(job, "[5/6] LLM loaded - generating chapters ...")

        chapters = []
        for i, ch in enumerate(merged):
            job["progress"] = f"5 / 6  ({i + 1}/{len(merged)})"
            log_msg(job, f"[5/6] Generating chapter {i + 1}/{len(merged)} [{ch['start_time']}] ...")
            ch_json = generate_chapter_json(ch, llm, tokenizer, i)
            ch_json["keywords"]        = ch.get("keywords", [])
            ch_json["boundary_source"] = "visual_cosine_sliding_window"
            log_msg(job, f"[5/6]   -> \"{ch_json['chapter']}\"")
            chapters.append(ch_json)

        log_msg(job, f"[5/6] Classifying {len(chapters)} chapters as key/skip ...")
        for i, ch in enumerate(chapters):
            ch["key"] = classify_high_level(ch, llm, tokenizer)
            log_msg(job, f"[5/6]   [{i + 1}/{len(chapters)}] {ch['chapter']} -> {'KEY' if ch['key'] == 'True' else 'skip'}")
            job["progress"] = f"5 / 6  ({i + 1}/{len(chapters)})"

        del llm, tokenizer

        # ── 6  Save ───────────────────────────────────────────────────────────
        job["progress"] = "6 / 6"
        out_path = os.path.join(CHAPTER_DIR, f"{stem}-chapters.json")
        log_msg(job, f"[6/6] Saving {len(chapters)} chapters to {out_path} ...")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(chapters, f, indent=2, ensure_ascii=False)

        log_msg(job, "[6/6] Done! Chapter index written successfully.")
        job.update(status="done", step="Pipeline complete!", progress="6 / 6")

    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"
        log_msg(job, f"ERROR: {err}")
        job.update(status="error", error=err, step="Pipeline failed")
        traceback.print_exc()


# ═══════════════════════════════════════════════════════════════
#  Flask routes
# ═══════════════════════════════════════════════════════════════
@app.route("/")
def index():
    videos = get_available_videos()
    return render_template(
        "UI-template.html",
        chapters_json="[]",
        total_duration=0,
        video_path="",
        available_videos=videos,
        current_video=None,
        has_chapters=False,
    )


@app.route("/video/<video_name>")
def view_video(video_name):
    vfile = os.path.join(UPLOAD_FOLDER, video_name)
    if not os.path.exists(vfile):
        return redirect("/")

    chapters = load_chapters(video_name)
    enriched = enrich_chapters(chapters) if chapters else []
    total    = max((c["end_sec"] for c in enriched), default=0)
    videos   = get_available_videos()

    return render_template(
        "UI-template.html",
        chapters_json=json.dumps(enriched),
        total_duration=total,
        video_path=url_for("static", filename=f"videos/{video_name}"),
        available_videos=videos,
        current_video=video_name,
        has_chapters=len(chapters) > 0,
    )


@app.route("/upload", methods=["POST"])
def upload():
    if "video" not in request.files:
        return jsonify(status="error", message="No file provided"), 400
    f = request.files["video"]
    if not f.filename:
        return jsonify(status="error", message="Empty filename"), 400
    ext = Path(f.filename).suffix.lower()
    if ext not in ALLOWED_EXT:
        return jsonify(status="error", message=f"Unsupported format: {ext}"), 400

    safe = f.filename.replace(" ", "_")
    dest = os.path.join(UPLOAD_FOLDER, safe)
    f.save(dest)

    jid = uuid.uuid4().hex[:8]
    jobs[jid] = dict(status="running", step="Uploaded - queuing pipeline ...", progress="0 / 6", error=None, logs=[])
    threading.Thread(target=run_pipeline, args=(jid, dest, safe), daemon=True).start()
    return jsonify(status="ok", job_id=jid, video_name=safe)


@app.route("/process/<video_name>", methods=["POST"])
def process_existing(video_name):
    vpath = os.path.join(UPLOAD_FOLDER, video_name)
    if not os.path.exists(vpath):
        return jsonify(status="error", message="Video not found"), 404

    jid = uuid.uuid4().hex[:8]
    jobs[jid] = dict(status="running", step="Starting pipeline ...", progress="0 / 6", error=None, logs=[])
    threading.Thread(target=run_pipeline, args=(jid, vpath, video_name), daemon=True).start()
    return jsonify(status="ok", job_id=jid, video_name=video_name)


@app.route("/api/job/<job_id>")
def job_status(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify(status="error", error="Unknown job"), 404
    return jsonify(job)


# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("AutoChapter — Visual Sliding-Window Pipeline")
    print("Running at http://localhost:5000")
    app.run(debug=True, use_reloader=False)
