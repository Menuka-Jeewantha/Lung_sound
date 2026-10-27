"""
Lung Sound Classification System - Streamlit app
Stage 1: Binary model  -> Healthy vs Abnormal (lung_sound_binary_classification_model.h5)
Stage 2: Ensemble model -> 9-class respiratory disease classification (ensemble_model.h5)
"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tempfile
from pathlib import Path

import librosa
import librosa.display
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model

# --------------------------------------------------------------------------
# Constants (matching your training pipeline)
# --------------------------------------------------------------------------
N_FFT = 2048
HOP_LENGTH = 512
N_MFCC = 128
FIXED_LENGTH = 264
INPUT_SHAPE = (N_MFCC, FIXED_LENGTH, 3)

SEGMENT_SAMPLES = (FIXED_LENGTH - 1) * HOP_LENGTH

CLASS_NAMES = [
    "Asthma",
    "Bronchiectasis",
    "Bronchiolitis",
    "Bronchitis",
    "COPD",
    "Lung Fibrosis",
    "Pleural Effusion",
    "Pneumonia",
    "URTI",
]

APP_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Lung Sound Classifier",
    page_icon="🫁",
    layout="wide",
)


# --------------------------------------------------------------------------
# Model loading
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading models...")
def load_models():
  possible_dirs = [APP_DIR, APP_DIR / "models"]

  binary_path, ensemble_path = None, None
  for d in possible_dirs:
    b_candidate = d / "lung_sound_binary_classification_model.h5"
    e_candidate = d / "ensemble_model.h5"
    if b_candidate.exists():
      binary_path = b_candidate
    if e_candidate.exists():
      ensemble_path = e_candidate

  if not binary_path or not ensemble_path:
    raise FileNotFoundError(
        "Could not find 'lung_sound_binary_classification_model.h5' or"
        " 'ensemble_model.h5' in the project directory."
    )

  binary_model = load_model(binary_path, compile=False)
  ensemble_model = load_model(ensemble_path, compile=False)
  return binary_model, ensemble_model


# --------------------------------------------------------------------------
# Audio processing & features
# --------------------------------------------------------------------------
def load_audio(uploaded_file):
  suffix = Path(uploaded_file.name).suffix or ".wav"
  with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
    tmp.write(uploaded_file.getvalue())
    tmp_path = tmp.name
  try:
    y, sr = librosa.load(tmp_path, sr=None)
  finally:
    os.remove(tmp_path)
  return y, sr


def peak_normalize(y):
  peak = float(np.max(np.abs(y))) if len(y) else 0.0
  return y / peak if peak > 0 else y


def split_segments(y, max_segments):
  if len(y) <= SEGMENT_SAMPLES:
    return [y]
  n_full = len(y) // SEGMENT_SAMPLES
  segs = [
      y[i * SEGMENT_SAMPLES : (i + 1) * SEGMENT_SAMPLES] for i in range(n_full)
  ]
  tail = y[n_full * SEGMENT_SAMPLES :]
  if len(tail) >= SEGMENT_SAMPLES // 2:
    segs.append(tail)
  return segs[:max_segments]


def extract_features(y, sr):
  mfcc = librosa.feature.mfcc(
      y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH
  )
  mfcc = librosa.util.fix_length(mfcc, size=FIXED_LENGTH, axis=1)

  mel = librosa.feature.melspectrogram(
      y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH
  )
  mel = librosa.power_to_db(mel, ref=np.max)
  mel = librosa.util.fix_length(mel, size=FIXED_LENGTH, axis=1)
  mel = np.resize(mel, (N_MFCC, FIXED_LENGTH))

  chroma = librosa.feature.chroma_stft(
      y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH
  )
  chroma = librosa.util.fix_length(chroma, size=FIXED_LENGTH, axis=1)
  chroma = np.resize(chroma, (N_MFCC, FIXED_LENGTH))

  return np.stack((mfcc, mel, chroma), axis=-1).astype("float32")


def predict_binary(model, X, abnormal_is_one):
  p = model.predict(X, verbose=0).reshape(-1)
  p_abnormal = p if abnormal_is_one else 1.0 - p
  return float(np.mean(p_abnormal))


def predict_ensemble(model, X):
  preds = model.predict(X, verbose=0)
  return preds.mean(axis=0)


# --------------------------------------------------------------------------
# Plotting Helpers
# --------------------------------------------------------------------------
def plot_probabilities(probs):
  order = np.argsort(probs)
  fig, ax = plt.subplots(figsize=(7, 4))
  colors = ["#c9d6e8"] * len(order)
  colors[-1] = "#1f6feb"
  ax.barh([CLASS_NAMES[i] for i in order], probs[order] * 100, color=colors)
  for y_pos, i in enumerate(order):
    ax.text(
        probs[i] * 100 + 1,
        y_pos,
        f"{probs[i] * 100:.1f}%",
        va="center",
        fontsize=9,
    )
  ax.set_xlim(0, 110)
  ax.set_xlabel("Probability (%)")
  ax.spines[["top", "right"]].set_visible(False)
  fig.tight_layout()
  return fig


def plot_audio_views(y, sr, features):
  fig, axes = plt.subplots(1, 3, figsize=(16, 3.6))
  librosa.display.waveshow(y, sr=sr, ax=axes[0], color="#1f6feb")
  axes[0].set_title("Waveform")
  librosa.display.specshow(
      features[:, :, 1],
      sr=sr,
      hop_length=HOP_LENGTH,
      x_axis="time",
      y_axis="mel",
      ax=axes[1],
  )
  axes[1].set_title("Mel spectrogram (dB)")
  librosa.display.specshow(
      features[:, :, 0], sr=sr, hop_length=HOP_LENGTH, x_axis="time", ax=axes[2]
  )
  axes[2].set_title("MFCC")
  fig.tight_layout()
  return fig


# --------------------------------------------------------------------------
# UI Setup
# --------------------------------------------------------------------------
with st.sidebar:
  st.header("Settings")
  abnormal_is_one = (
      st.radio(
          "Binary model output = 1 means", ["Abnormal", "Healthy"], index=0
      )
      == "Abnormal"
  )
  threshold = st.slider("Abnormal threshold", 0.05, 0.95, 0.50, 0.05)
  always_classify = st.checkbox(
      "Always run disease classifier",
      value=False,
      help="Run the 9-class ensemble model even if the sound is screened as healthy.",
  )
  normalize_audio = st.checkbox("Peak-normalise audio", value=True)
  use_segments = st.checkbox("Analyse long recordings in windows", value=True)
  max_segments = st.slider("Max windows", 1, 30, 10, disabled=not use_segments)

st.title("🫁 Lung Sound Classification System")
st.caption("Stage 1: Binary Screening → Stage 2: 9-Class Ensemble Model")
st.info(
    "Research/educational prototype. It is **not** a medical device and must not"
    " be used for diagnosis or treatment decisions."
)

# Load models
try:
  binary_model, ensemble_model = load_models()
except Exception as exc:
  st.error(f"Failed to load models: {exc}")
  st.stop()

uploaded = st.file_uploader(
    "Upload a lung sound recording", type=["wav", "mp3", "flac", "ogg"]
)

if uploaded is None:
  st.stop()

st.audio(uploaded.getvalue())

if not st.button("Analyse", type="primary"):
  st.stop()

with st.spinner("Extracting features and running models..."):
  try:
    y, sr = load_audio(uploaded)
  except Exception as exc:
    st.error(f"Could not read this audio file: {exc}")
    st.stop()

  if len(y) < N_FFT:
    st.error("Recording is too short to analyse.")
    st.stop()

  if normalize_audio:
    y = peak_normalize(y)

  segments = split_segments(y, max_segments) if use_segments else [y]
  X = np.stack([extract_features(seg, sr) for seg in segments])

  p_abnormal = predict_binary(binary_model, X, abnormal_is_one)
  is_abnormal = p_abnormal >= threshold

  ens_probs = None
  if is_abnormal or always_classify:
    ens_probs = predict_ensemble(ensemble_model, X)

# Display Results
st.divider()
st.subheader("Results")
c1, c2, c3 = st.columns(3)
c1.metric(
    "Duration",
    f"{len(y) / sr:.1f} s",
    help=f"Sample rate {sr} Hz · {len(segments)} window(s)",
)
c2.metric("Screening", "Abnormal" if is_abnormal else "Healthy")
c3.metric("P(abnormal)", f"{p_abnormal * 100:.1f}%")
st.progress(min(max(p_abnormal, 0.0), 1.0))

if not is_abnormal and ens_probs is None:
  st.success("No respiratory abnormality detected by the binary model.")
elif ens_probs is not None:
  top = int(np.argmax(ens_probs))
  if is_abnormal:
    st.warning(
        f"Abnormal lung sound detected. Most likely condition:"
        f" **{CLASS_NAMES[top]}** ({ens_probs[top] * 100:.1f}%)"
    )
  else:
    st.info(
        f"Sound screened as healthy; disease classifier leans towards:"
        f" **{CLASS_NAMES[top]}** ({ens_probs[top] * 100:.1f}%)"
    )

  if ens_probs[top] < 0.5:
    st.caption(
        "⚠️ Low confidence: no single class dominates. Treat this prediction"
        " with caution."
    )

  left, right = st.columns([3, 2])
  with left:
    fig = plot_probabilities(ens_probs)
    st.pyplot(fig)
    plt.close(fig)
  with right:
    st.markdown("**Top 3 Predictions**")
    for i in np.argsort(ens_probs)[::-1][:3]:
      st.write(f"{CLASS_NAMES[i]} — {ens_probs[i] * 100:.1f}%")

with st.expander("Audio & feature visualisation (first window)"):
  fig = plot_audio_views(segments[0], sr, X[0])
  st.pyplot(fig)
  plt.close(fig)