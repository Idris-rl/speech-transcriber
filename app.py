from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import whisper
import tempfile
import os
import sys

app = Flask(__name__)
CORS(app)

# Load Whisper model on startup
model = whisper.load_model("tiny")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file"}), 400

    audio = request.files["audio"]
    suffix = ".webm"
    if audio.filename and audio.filename.endswith(".wav"):
        suffix = ".wav"

    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    audio.save(tmp.name)
    tmp.close()

    try:
        result = model.transcribe(tmp.name)
        text = result["text"]
        return jsonify({"text": text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        os.unlink(tmp.name)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
