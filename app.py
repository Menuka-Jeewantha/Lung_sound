import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import tempfile
# Replace: from tensorflow.keras.models import load_model
from tf_keras.models import load_model

# -----------------------------------------------------------------------------
# Page Configuration & Custom Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Lung Sound Classification System",
    page_icon="🫁",
    layout="wide"
)

st.title("🫁 Automated Lung Sound Classification System")
st.markdown("""
This application utilizes Deep Learning and Audio Signal Processing to classify respiratory sounds recorded from the ICBHI 2017 dataset.
It processes audio features (**MFCC**, **Mel Spectrogram**, **Chroma STFT**) and computes a weighted ensemble prediction using a **Hybrid Model (60%)** and a **CNN Model (40%)**.
""")

# -----------------------------------------------------------------------------
# Global Constants & Configuration
# -----------------------------------------------------------------------------
N_FFT = 2048
HOP_LENGTH = 512
N_MFCC = 128
FIXED_LENGTH = 264

CLASS_NAMES = [
    'Asthma', 
    'Bronchiectasis', 
    'Bronchiolitis', 
    'Bronchitis',
    'COPD', 
    'Lung Fibrosis', 
    'Pleural Effusion', 
    'Pneumonia', 
    'URTI'
]

WEIGHT_HYBRID = 0.6
WEIGHT_CNN = 0.4

# -----------------------------------------------------------------------------
# Model Loading (Cached)
# -----------------------------------------------------------------------------
@st.cache_resource
def load_models():
    hybrid_path = 'lung_sound_Hybrid_multiclass_classification_model.h5'
    cnn_path = 'lung_sound_multiclass_classification_model.h5'
    
    hybrid_model = None
    cnn_model = None
    
    if os.path.exists(hybrid_path):
        hybrid_model = load_model(hybrid_path, compile=False)
    else:
        st.warning(f"⚠️ Hybrid model file (`{hybrid_path}`) not found. Please upload or place it in the working directory.")
        
    if os.path.exists(cnn_path):
        cnn_model = load_model(cnn_path, compile=False)
    else:
        st.warning(f"⚠️ CNN model file (`{cnn_path}`) not found. Please upload or place it in the working directory.")
        
    return hybrid_model, cnn_model

# -----------------------------------------------------------------------------
# Feature Extraction Function
# -----------------------------------------------------------------------------
def extract_features(audio_file_path):
    # Load audio file with original sampling rate
    y, sr = librosa.load(audio_file_path, sr=None)
    
    # 1. MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
    mfcc = librosa.util.fix_length(mfcc, size=FIXED_LENGTH, axis=1)
    
    # 2. Mel Spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
    mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    mel_spectrogram = librosa.util.fix_length(mel_spectrogram, size=FIXED_LENGTH, axis=1)
    mel_spectrogram = np.resize(mel_spectrogram, (N_MFCC, FIXED_LENGTH))
    
    # 3. Chroma STFT
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
    chroma_stft = librosa.util.fix_length(chroma_stft, size=FIXED_LENGTH, axis=1)
    chroma_stft = np.resize(chroma_stft, (N_MFCC, FIXED_LENGTH))
    
    # Stack features into 3D representation (128, 264, 3)
    features_3d = np.stack((mfcc, mel_spectrogram, chroma_stft), axis=-1)
    return y, sr, mfcc, mel_spectrogram, chroma_stft, features_3d

# -----------------------------------------------------------------------------
# Main Interface & File Upload
# -----------------------------------------------------------------------------
hybrid_model, cnn_model = load_models()

st.sidebar.header("🕹️ Model Settings & Info")
st.sidebar.markdown(f"""
**Ensemble Model Weights:**
- **Hybrid Model Weight:** {WEIGHT_HYBRID}
- **CNN Model Weight:** {WEIGHT_CNN}

**Target Respiratory Conditions ({len(CLASS_NAMES)}):**
- Asthma
- Bronchiectasis
- Bronchiolitis
- Bronchitis
- COPD
- Lung Fibrosis
- Pleural Effusion
- Pneumonia
- URTI (Upper Respiratory Tract Infection)
""")

uploaded_file = st.file_uploader("📂 Upload a Lung Sound Audio File (.wav)", type=["wav", "mp3", "ogg"])

if uploaded_file is not None:
    # Save uploaded file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_audio_path = tmp_file.name

    st.subheader("🔊 Audio Player & Feature Visualizations")
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.audio(temp_audio_path, format="audio/wav")
        st.info("Uploaded Audio Sample successfully loaded.")

    # Extract Features
    with st.spinner("Extracting audio features (MFCC, Mel Spectrogram, Chroma)..."):
        y, sr, mfcc, mel_spec, chroma, features_3d = extract_features(temp_audio_path)

    # Feature Visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["Waveform", "MFCC", "Mel Spectrogram", "Chroma STFT"])
    
    with tab1:
        fig, ax = plt.subplots(figsize=(10, 3))
        librosa.display.waveshow(y, sr=sr, ax=ax, color='crimson')
        ax.set_title("Raw Audio Waveform")
        st.pyplot(fig)
        plt.close()

    with tab2:
        fig, ax = plt.subplots(figsize=(10, 3))
        img = librosa.display.specshow(mfcc, x_axis='time', sr=sr, hop_length=HOP_LENGTH, ax=ax)
        fig.colorbar(img, ax=ax)
        ax.set_title("MFCC (128 Coefficients)")
        st.pyplot(fig)
        plt.close()

    with tab3:
        fig, ax = plt.subplots(figsize=(10, 3))
        img = librosa.display.specshow(mel_spec, x_axis='time', y_axis='mel', sr=sr, hop_length=HOP_LENGTH, ax=ax)
        fig.colorbar(img, ax=ax, format="%+2.0f dB")
        ax.set_title("Mel Spectrogram (dB)")
        st.pyplot(fig)
        plt.close()

    with tab4:
        fig, ax = plt.subplots(figsize=(10, 3))
        img = librosa.display.specshow(chroma, x_axis='time', y_axis='chroma', sr=sr, hop_length=HOP_LENGTH, ax=ax)
        fig.colorbar(img, ax=ax)
        ax.set_title("Chroma STFT")
        st.pyplot(fig)
        plt.close()

    # Model Inference Section
    st.divider()
    st.subheader("🔍 Disease Classification Results")

    if hybrid_model is None or cnn_model is None:
        st.error("❌ Models could not be loaded. Please ensure `lung_sound_Hybrid_multiclass_classification_model.h5` and `lung_sound_multiclass_classification_model.h5` are placed in the application folder.")
    else:
        # Prepare batch dimension for input: (1, 128, 264, 3)
        input_data = np.expand_dims(features_3d, axis=0)

        with st.spinner("Running ensemble inference..."):
            hybrid_pred = hybrid_model.predict(input_data)[0]
            cnn_pred = cnn_model.predict(input_data)[0]

            # Weighted Average Ensemble Prediction
            ensemble_pred = (WEIGHT_HYBRID * hybrid_pred) + (WEIGHT_CNN * cnn_pred)
            predicted_class_idx = np.argmax(ensemble_pred)
            predicted_disease = CLASS_NAMES[predicted_class_idx]
            confidence = ensemble_pred[predicted_class_idx] * 100

        # Show Primary Result
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.metric("Predicted Disease Condition", predicted_disease)
        with res_col2:
            st.metric("Ensemble Confidence Score", f"{confidence:.2f}%")

        # Class Probabilities Breakdown
        st.markdown("### 📊 Ensemble Class Probability Breakdown")
        
        # Sort classes by probability for nicer presentation
        prob_dict = {CLASS_NAMES[i]: float(ensemble_pred[i]) for i in range(len(CLASS_NAMES))}
        sorted_probs = dict(sorted(prob_dict.items(), key=lambda item: item[1], reverse=True))

        for disease, prob in sorted_probs.items():
            st.write(f"**{disease}**: {prob * 100:.2f}%")
            st.progress(float(prob))

        # Show Individual Model Contributions
        with st.expander("🔎 View Individual Model Probabilities"):
            m_col1, m_col2 = st.columns(2)
            with m_col1:
                st.markdown("**Hybrid Model Predictions (Weight: 0.6):**")
                for i, cname in enumerate(CLASS_NAMES):
                    st.write(f"- {cname}: {hybrid_pred[i]*100:.2f}%")
            with m_col2:
                st.markdown("**CNN Model Predictions (Weight: 0.4):**")
                for i, cname in enumerate(CLASS_NAMES):
                    st.write(f"- {cname}: {cnn_pred[i]*100:.2f}%")

    # Clean up temporary file
    if os.path.exists(temp_audio_path):
        os.remove(temp_audio_path)