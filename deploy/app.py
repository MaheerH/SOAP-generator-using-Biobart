# app.py
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig
import torch
from pathlib import Path

# -------------------------
# Model setup
# -------------------------
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig, GenerationConfig
import torch

MODEL_ID = "maheer007/biobart-medical-finetuned"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# 1) Load and patch the *model* config first
cfg = AutoConfig.from_pretrained(MODEL_ID)
if getattr(cfg, "early_stopping", None) is None:
    cfg.early_stopping = False  # or True / "never" if you prefer

# 2) Now load the model with the patched config
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID, config=cfg)

# 3) (Optional) also load/patch GenerationConfig used at generation time
gen_cfg = GenerationConfig.from_pretrained(MODEL_ID)
if gen_cfg.early_stopping is None:
    gen_cfg.early_stopping = False

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


# -------------------------
# FastAPI setup
# -------------------------
app = FastAPI(title="Medical Chat Summarization API")

# CORS (optional; handy if you call from other origins)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static & templates
BASE_DIR = Path(__file__).parent
static_dir = BASE_DIR / "static"
templates_dir = BASE_DIR / "templates"

if static_dir.exists():
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# -------------------------
# Pydantic model for POST /summarize
# -------------------------
class DialogueInput(BaseModel):
    dialogue: str
    max_new_tokens: int | None = 256
    min_new_tokens: int | None = 16
    temperature: float | None = 0.7
    top_p: float | None = 0.95

# -------------------------
# Routes
# -------------------------
@app.get("/", response_class=HTMLResponse)
def home_page():
    """Serve the simple UI."""
    html_path = templates_dir / "index.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))

@app.get("/health")
def health():
    return {"status": "ok", "message": "Medical Summarizer API"}

@app.post("/summarize")
def summarize(data: DialogueInput):
    text = (data.dialogue or "").strip()
    if not text:
        return JSONResponse({"error": "Empty dialogue"}, status_code=400)

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    ).to(device)

    outputs = model.generate(
        **inputs,
        generation_config=gen_cfg,
        do_sample=True,
        temperature=data.temperature or 0.7,
        top_p=data.top_p or 0.95,
        max_new_tokens=data.max_new_tokens or 256,
        min_new_tokens=data.min_new_tokens or 16,
        no_repeat_ngram_size=3,
    )

    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"summary": summary}
