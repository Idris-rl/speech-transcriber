import os, sys, tempfile, subprocess, traceback
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from faster_whisper import WhisperModel

app = Flask(__name__)
CORS(app)

# Check ffmpeg is available
try:
    subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
except:
    print("ERROR: ffmpeg not found!", file=sys.stderr)

model = WhisperModel("tiny", device="cpu", compute_type="int8")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/_ping")
def ping():
    return "ok"

@app.route("/_check")
def check():
    ffmpeg = "yes"
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except:
        ffmpeg = "no"
    return jsonify({"ffmpeg": ffmpeg, "status": "ok"})

@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file"}), 400

    audio = request.files["audio"]
    suffix = ".wav"
    if audio.filename and audio.filename.endswith(".webm"):
        suffix = ".webm"

    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        audio.save(tmp.name)
        tmp.close()

        size = os.path.getsize(tmp.name)
        if size == 0:
            return jsonify({"error": "Empty audio file"}), 400

        segments, info = model.transcribe(tmp.name, beam_size=1)
        text = " ".join(segment.text for segment in segments)
        return jsonify({"text": text, "duration": info.duration})
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500
    finally:
        try:
            os.unlink(tmp.name)
        except:
            pass

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
