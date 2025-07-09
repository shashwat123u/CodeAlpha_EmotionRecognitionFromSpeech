
import gradio as gr
import librosa
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

# Load your trained model
model = tf.keras.models.load_model("speech_emotion_crnn_model.keras")

# Label encoder (same order used in training)
label_encoder = LabelEncoder()
label_encoder.classes_ = np.array(['angry', 'disgust', 'fear', 'happy', 'neutral', 'pleasant_surprise', 'sad'])

# Prediction function
def predict_emotion(audio):
    if audio is None:
        return "No audio uploaded."
    sr, signal = audio
    signal = np.array(signal)

    mfccs = librosa.feature.mfcc(y=signal.astype(float), sr=sr, n_mfcc=40)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    X_input = mfccs_mean.reshape(1, 10, 4, 1)

    pred = model.predict(X_input)
    emotion = label_encoder.inverse_transform([np.argmax(pred)])
    return f"🎭 Emotion: {emotion[0]}"

# Gradio interface
gr.Interface(
    fn=predict_emotion,
    inputs=gr.Audio(type="numpy", label="📁 Upload a .wav file"),
    outputs=gr.Textbox(label="Predicted Emotion"),
    title="🎤 Speech Emotion Recognition",
    description="Upload a .wav file to detect emotion in speech",
    allow_flagging="never"
).launch(share=True)
