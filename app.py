from flask import Flask, render_template, request
import numpy as np
import librosa
import os
import webbrowser
from threading import Timer
from tensorflow.keras.models import load_model

# ✅ Reduce TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

app = Flask(__name__)

# ✅ Load model
model = load_model("models/bilstm_speakerwise.keras")


# ✅ Feature extraction (FIXED + NORMALIZED)
def extract_logmel_sequence(path, max_len=300):
    y, sr = librosa.load(path, duration=30)

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
    log_mel = librosa.power_to_db(mel).T

    # ✅ Normalize (VERY IMPORTANT FIX)
    log_mel = (log_mel - np.mean(log_mel)) / (np.std(log_mel) + 1e-6)

    # ✅ Padding / Truncation
    if len(log_mel) < max_len:
        pad = np.zeros((max_len - len(log_mel), 64))
        log_mel = np.vstack((log_mel, pad))
    else:
        log_mel = log_mel[:max_len]

    return log_mel


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # ✅ Validate file
            if 'audio' not in request.files:
                return "<h3>No file uploaded</h3><a href='/'>Go Back</a>"

            file = request.files['audio']

            if file.filename == '':
                return "<h3>No selected file</h3><a href='/'>Go Back</a>"

            # ✅ Save file
            filepath = "temp.wav"
            file.save(filepath)

            # ✅ Extract features
            features = extract_logmel_sequence(filepath)
            features = np.expand_dims(features, axis=0)

            # ✅ Prediction
            prediction = model.predict(features)[0]

            print("Raw Prediction:", prediction)  # DEBUG

            control_prob = float(prediction[0])
            dementia_prob = float(prediction[1])

            os.remove(filepath)

            # ✅ FIXED DECISION LOGIC (THRESHOLD BASED)
            if dementia_prob > 0.5:
                result = "⚠️ DEMENTIA DETECTED"
            else:
                result = "✅ CONTROL (Healthy Speech)"

            return render_template(
                'result.html',
                result=result,
                probs=[
                    round(control_prob, 4),
                    round(dementia_prob, 4)
                ]
            )

        except Exception as e:
            return f"<h2 style='color:red'>Error: {e}</h2><a href='/'>Try Again</a>"

    return render_template('index.html')


# ✅ Open browser only once
def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000")


if __name__ == "__main__":
    print("👉 http://127.0.0.1:5000")

    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        Timer(1, open_browser).start()

    app.run(host="127.0.0.1", port=5000, debug=True)
