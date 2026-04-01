"""
Granite 4.0 1B Speech — Local inference server for v-voice-claude.

Loads the IBM Granite speech model and serves an HTTP API for transcription.
Listens on http://127.0.0.1:8976

Endpoints:
  POST /transcribe  — multipart form with "file" (WAV) and optional "language"
                       Returns JSON { "text": "..." }
  GET  /health      — Returns { "status": "ready" } when model is loaded
  POST /shutdown    — Gracefully shuts down the server
"""

import argparse
import io
import os
import signal
import sys
import threading

import torch
import torchaudio
from flask import Flask, jsonify, request
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

app = Flask(__name__)

# Global model/processor handles
processor = None
tokenizer = None
model = None
device = None
model_loaded = threading.Event()


def load_model(model_dir: str):
    """Load the Granite speech model into memory."""
    global processor, tokenizer, model, device

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print(f"[granite] Loading model from {model_dir} on {device} ...")

    processor = AutoProcessor.from_pretrained(model_dir)
    tokenizer = processor.tokenizer
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_dir,
        device_map=device,
        torch_dtype=dtype,
    )
    model.eval()

    print("[granite] Model loaded and ready.")
    model_loaded.set()


@app.route("/health", methods=["GET"])
def health():
    if model_loaded.is_set():
        return jsonify({"status": "ready"})
    else:
        return jsonify({"status": "loading"}), 503


@app.route("/transcribe", methods=["POST"])
def transcribe():
    if not model_loaded.is_set():
        return jsonify({"error": "Model not loaded yet"}), 503

    # Read uploaded audio file
    if "file" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files["file"]
    audio_bytes = audio_file.read()

    language = request.form.get("language", "en")

    try:
        # Load audio from bytes
        wav, sr = torchaudio.load(io.BytesIO(audio_bytes))

        # Ensure mono
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        # Resample to 16kHz if needed
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            wav = resampler(wav)

        # Build prompt
        user_prompt = "<|audio|>can you transcribe the speech into a written format?"
        chat = [{"role": "user", "content": user_prompt}]
        prompt = tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )

        # Run inference
        model_inputs = processor(prompt, wav, device=device, return_tensors="pt").to(
            device
        )

        with torch.no_grad():
            model_outputs = model.generate(
                **model_inputs,
                max_new_tokens=500,
                do_sample=False,
                num_beams=1,
            )

        # Decode — strip input tokens
        num_input_tokens = model_inputs["input_ids"].shape[-1]
        new_tokens = model_outputs[0, num_input_tokens:].unsqueeze(0)
        output_text = tokenizer.batch_decode(
            new_tokens, add_special_tokens=False, skip_special_tokens=True
        )

        text = output_text[0].strip() if output_text else ""
        return jsonify({"text": text})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/shutdown", methods=["POST"])
def shutdown():
    """Gracefully stop the server."""
    os.kill(os.getpid(), signal.SIGTERM)
    return jsonify({"status": "shutting down"})


def main():
    parser = argparse.ArgumentParser(description="Granite Speech inference server")
    parser.add_argument(
        "--model-dir",
        required=True,
        help="Path to the downloaded Granite model directory",
    )
    parser.add_argument("--port", type=int, default=8976, help="Server port")
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    args = parser.parse_args()

    # Load model in background thread so server starts responding quickly
    loader_thread = threading.Thread(
        target=load_model, args=(args.model_dir,), daemon=True
    )
    loader_thread.start()

    print(f"[granite] Server starting on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
