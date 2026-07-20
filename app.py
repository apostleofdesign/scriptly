import os
from flask import Flask, request, render_template, jsonify
from faster_whisper import WhisperModel

app = Flask(__name__)
UPLOAD_FOLDER = "/tmp/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

print("Loading model...")
model = WhisperModel("tiny", device="cpu", compute_type="int8")
print("Model loaded.")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "video" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["video"]
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)
    segments, info = model.transcribe(filepath)
    text = " ".join(seg.text.strip() for seg in segments)
    os.remove(filepath)
    return jsonify({"transcript": text})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
