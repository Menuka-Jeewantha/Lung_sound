import os
import librosa
import numpy as np
import streamlit as tf_st
import streamlit as st
from tensorflow.keras.models import load_model

# Page configuration
st.set_page_config(
    page_title="Lung Sound Classification System", page_icon="🫁", layout="centered"
)

# Feature extraction parameters matching your training pipeline
n_fft = 2048
hop_length = 512
n_mfcc = 128
fixed_length = 264

class_names = [
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


@st.cache_resource
def load_trained_models():
  # Load models with compile=False to avoid optimizer state mismatches during inference
  hybrid_model = load_model(
      "lung_sound_Hybrid_multiclass_classification_model.h5", compile=False
  )
  cnn_model = load_model(
      "lung_sound_multiclass_classification_model.h5", compile=False
  )
  return hybrid_model, cnn_model


hybrid_model, cnn_model = load_trained_models()


def extract_features(audio_file):
  y, sr = librosa.load(audio_file, sr=None)

  # MFCC
  mfcc = librosa.feature.mfcc(
      y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length
  )
  mfcc = librosa.util.fix_length(mfcc, size=fixed_length, axis=1)

  # Mel Spectrogram
  mel_spectrogram = librosa.feature.melspectrogram(
      y=y, sr=sr, n_fft=n_fft, hop_length=hop_length
  )
  mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
  mel_spectrogram = librosa.util.fix_length(
      mel_spectrogram, size=fixed_length, axis=1
  )
  mel_spectrogram = np.resize(mel_spectrogram, (n_mfcc, fixed_length))

  # Chroma STFT
  chroma_stft = librosa.feature.chroma_stft(
      y=y, sr=sr, n_fft=n_fft, hop_length=hop_length
  )
  chroma_stft = librosa.util.fix_length(chroma_stft, size=fixed_length, axis=1)
  chroma_stft = np.resize(chroma_stft, (n_mfcc, fixed_length))

  # Stack features into 3D array
  features_3d = np.stack((mfcc, mel_spectrogram, chroma_stft), axis=-1)
  return features_3d


def weighted_average_ensemble(
    input_data, weight_hybrid=0.6, weight_cnn=0.4
):
  hybrid_pred = hybrid_model.predict(input_data)
  cnn_pred = cnn_model.predict(input_data)
  ensemble_pred = (weight_hybrid * hybrid_pred) + (weight_cnn * cnn_pred)
  return ensemble_pred


# Streamlit Interface
st.title("🫁 Lung Sound Classification System")
st.markdown(
    "Upload a respiratory audio recording (`.wav`) to analyze and classify"
    " potential conditions using your weighted ensemble deep learning model."
)

uploaded_file = st.file_uploader(
    "Choose a lung sound audio file (.wav)", type=["wav"]
)

if uploaded_file is not None:
  st.audio(uploaded_file, format="audio/wav")

  if st.button("Run Classification"):
    with st.spinner("Extracting audio features and running model inference..."):
      # Save uploaded file temporarily
      temp_path = "temp_audio.wav"
      with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

      try:
        # Extract features and format input shape
        features = extract_features(temp_path)
        input_data = np.expand_dims(features, axis=0)

        # Predict using ensemble
        ensemble_preds = weighted_average_ensemble(
            input_data, weight_hybrid=0.6, weight_cnn=0.4
        )
        predicted_idx = np.argmax(ensemble_preds, axis=1)[0]
        predicted_label = class_names[predicted_idx]
        confidence = float(np.max(ensemble_preds) * 100)

        st.success("Analysis Complete!")
        st.metric(
            label="Predicted Respiratory Condition",
            value=predicted_label,
            delta=f"{confidence:.2f}% Confidence",
        )

        # Show distribution across all classes
        st.subheader("Class Probability Breakdown")
        probs = ensemble_preds[0]
        prob_dict = {
            class_names[i]: float(probs[i]) for i in range(len(class_names))
        }
        st.bar_chart(prob_dict)

      except Exception as e:
        st.error(f"An error occurred during feature extraction or prediction: {e}")

      finally:
        if os.path.exists(temp_path):
          os.remove(temp_path)