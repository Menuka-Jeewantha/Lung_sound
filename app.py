"""
Lung Sound Classification System - Streamlit app
ICBHI 2017 | Binary (healthy vs abnormal) + weighted ensemble (CNN + CNN-LSTM hybrid)

Expected model files (same folder as app.py, or in a ./models sub-folder):
    lung_sound_binary_classification_model.h5
    lung_sound_multiclass_classification_model.h5          (CNN, 9 classes)
    lung_sound_Hybrid_multiclass_classification_model.h5   (CNN-LSTM hybrid, 9 classes)

ensemble_model.h5 is NOT loaded: it is a subclassed Keras model that only stores its
class name, so the app rebuilds the weighted average (0.6 hybrid + 0.4 CNN) itself.
"""
import io
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# Constants - MUST match the training notebook exactly
N_FFT = 2048
HOP_LENGTH = 512
N_MFCC = 128
FIXED_LENGTH = 264
SEGMENT_SAMPLES = (FIXED_LENGTH - 1) * HOP_LENGTH

CLASS_NAMES = [
    "Asthma", "Bronchiectasis", "Bronchiolitis", "Bronchitis", "COPD",
    "Lung Fibrosis", "Pleural Effusion", "Pneumonia", "URTI",
]

BASE_DIR = Path(__file__).resolve().parent
MODEL_FILES = {
    "binary": "lung_sound_binary_classification_model.h5",
    "cnn": "lung_sound_multiclass_classification_model.h5",
    "hybrid": "lung_sound_Hybrid_multiclass_classification_model.h5",
}

st.set_page_config(page_title="Lung Sound Classifier", page_icon="🫁", layout="wide")


def _find_model(filename: str) -> Path:
    for folder in (BASE_DIR, BASE_DIR / "models"):
        p = folder / filename
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Could not find '{filename}' next to app.py or in a 'models/' folder."
    )


@st.cache_resource(show_spinner="Loading models...")
def load_models():
    import tensorflow as tf

    return {
        key: tf.keras.models.load_model(_find_model(fname), compile=False)
        for key, fname in MODEL_FILES.items()
    }


def extract_features_from_signal(y: np.ndarray, sr: int) -> np.ndarray:
    """Return a (128, 264, 3) array: [MFCC, Mel-spectrogram (dB), Chroma]."""
    if len(y) < N_FFT:
        y = np.pad(y, (0, N_FFT - len(y)))

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
    mfcc = librosa.util.fix_length(mfcc, size=FIXED_LENGTH, axis=1)

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
    mel = librosa.power_to_db(mel, ref=np.max)
    mel = librosa.util.fix_length(mel, size=FIXED_LENGTH, axis=1)
    mel = np.resize(mel, (N_MFCC, FIXED_LENGTH))

    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
    chroma = librosa.util.fix_length(chroma, size=FIXED_LENGTH, axis=1)
    chroma = np.resize(chroma, (N_MFCC, FIXED_LENGTH))

    return np.stack((mfcc, mel, chroma), axis=-1).astype("float32")


def split_into_segments(y: np.ndarray, max_segments: int) -> list:
    if len(y) <= SEGMENT_SAMPLES:
        return [y]
    segments = []
    for start in range(0, len(y), SEGMENT_SAMPLES):
        chunk = y[start:start + SEGMENT_SAMPLES]
        if len(chunk) < SEGMENT_SAMPLES // 2 and segments:
            break
        segments.append(chunk)
        if len(segments) >= max_segments:
            break
    return segments


def predict(models, batch, w_hybrid, abnormal_is_high, threshold):
    p_bin = models["binary"].predict(batch, verbose=0).reshape(-1)
    p_abnormal = float(np.mean(p_bin if abnormal_is_high else 1.0 - p_bin))

    hybrid_p = models["hybrid"].predict(batch, verbose=0)
    cnn_p = models["cnn"].predict(batch, verbose=0)
    ens = w_hybrid * hybrid_p + (1.0 - w_hybrid) * cnn_p
    return {
        "p_abnormal": p_abnormal,
        "is_abnormal": p_abnormal >= threshold,
        "ensemble": ens.mean(axis=0),
        "hybrid": hybrid_p.mean(axis=0),
        "cnn": cnn_p.mean(axis=0),
    }


st.title("🫁 Lung Sound Classification System")
st.caption(
    "ICBHI 2017 · Stage 1: healthy vs abnormal (CNN) · "
    "Stage 2: disease classification (weighted ensemble of CNN + CNN-LSTM hybrid)"
)

with st.sidebar:
    st.header("Settings")
    w_hybrid = st.slider("Ensemble weight - Hybrid model", 0.0, 1.0, 0.6, 0.05,
                         help="CNN weight = 1 - this value. Training notebook used 0.6 / 0.4.")
    st.write(f"CNN weight: **{1 - w_hybrid:.2f}**")
    threshold = st.slider("Abnormal threshold", 0.05, 0.95, 0.5, 0.05)
    abnormal_is_high = st.radio(
        "Binary model: sigmoid output close to 1 means...",
        ["Abnormal", "Healthy"], index=0,
        help="Depends on how you encoded labels when training the binary model. "
             "Flip this if healthy and abnormal results look reversed.",
    ) == "Abnormal"
    peak_norm = st.checkbox("Peak-normalise audio", value=True,
                            help="Training files were 'Normalized'. Turn off if your "
                                 "recordings are already normalised the same way.")
    use_segments = st.checkbox("Analyse long recordings in segments", value=True,
                               help="Models expect ~264 frames. Long files are split and "
                                    "predictions averaged.")
    max_segments = st.slider("Max segments", 1, 20, 8, disabled=not use_segments)
    always_disease = st.checkbox("Always show disease probabilities", value=False,
                                 help="By default the disease stage only runs for abnormal results.")

try:
    models = load_models()
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()

uploaded = st.file_uploader("Upload a lung sound recording", type=["wav", "flac", "ogg"])

if uploaded is None:
    st.info("Upload a `.wav` lung sound recording to begin.")
    st.stop()

st.audio(uploaded)

try:
    y, sr = librosa.load(io.BytesIO(uploaded.getvalue()), sr=None)
except Exception as e:
    st.error(f"Could not read the audio file: {e}")
    st.stop()

if len(y) == 0:
    st.error("The audio file is empty.")
    st.stop()

if peak_norm:
    y = librosa.util.normalize(y)

segments = split_into_segments(y, max_segments) if use_segments else [y[:SEGMENT_SAMPLES]]

with st.spinner("Extracting features and running models..."):
    batch = np.stack([extract_features_from_signal(s, sr) for s in segments])
    out = predict(models, batch, w_hybrid, abnormal_is_high, threshold)

c1, c2 = st.columns([1, 1])
with c1:
    st.subheader("Stage 1 - Healthy vs Abnormal")
    if out["is_abnormal"]:
        st.error(f"⚠️ Abnormal lung sound detected  ({out['p_abnormal'] * 100:.1f}% abnormal)")
    else:
        st.success(f"✅ Healthy lung sound  ({(1 - out['p_abnormal']) * 100:.1f}% healthy)")
    st.progress(min(max(out["p_abnormal"], 0.0), 1.0), text="Probability of abnormal")
    st.caption(f"{len(segments)} segment(s) analysed · sample rate {sr} Hz · "
               f"duration {len(y) / sr:.1f} s")

with c2:
    st.subheader("Stage 2 - Disease classification")
    if out["is_abnormal"] or always_disease:
        ens = out["ensemble"]
        top = int(np.argmax(ens))
        st.metric("Most likely condition", CLASS_NAMES[top], f"{ens[top] * 100:.1f}% confidence")
        df = pd.DataFrame({"Ensemble": ens, "Hybrid": out["hybrid"], "CNN": out["cnn"]},
                          index=CLASS_NAMES)
        st.bar_chart(df["Ensemble"])
    else:
        st.write("Skipped because the recording was classified as healthy.")

if out["is_abnormal"] or always_disease:
    with st.expander("Per-model probabilities"):
        st.dataframe((df * 100).round(2).style.format("{:.2f}%").highlight_max(axis=0, color="#2e7d32"),
                     use_container_width=True)

with st.expander("Audio visualisation"):
    fig, ax = plt.subplots(1, 2, figsize=(12, 3))
    librosa.display.waveshow(y, sr=sr, ax=ax[0])
    ax[0].set_title("Waveform")
    img = librosa.display.specshow(batch[0, :, :, 1], sr=sr, hop_length=HOP_LENGTH,
                                   x_axis="time", y_axis="mel", ax=ax[1])
    ax[1].set_title("Mel spectrogram (first segment, dB)")
    fig.colorbar(img, ax=ax[1], format="%+2.0f dB")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

st.divider()
st.caption(
    "⚕️ Research prototype for educational purposes only. It is not a medical device and must not "
    "be used for diagnosis or treatment decisions. Consult a qualified clinician."
)