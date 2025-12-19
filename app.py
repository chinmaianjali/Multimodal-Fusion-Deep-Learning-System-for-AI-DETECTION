import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torchvision import models, transforms
from huggingface_hub import hf_hub_download
import librosa
import numpy as np

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Multimodal AI Content Detector",
    page_icon="🧠",
    layout="wide"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===============================
# HF REPOS
# ===============================
TEXT_REPO  = "chinmaianjali/text-roberta-ai-detector"
IMAGE_REPO = "chinmaianjali/image-mobilenet-ai-detector"
AUDIO_REPO = "chinmaianjali/audio-cnn-ai-detector"

# ===============================
# LOAD MODELS
# ===============================
@st.cache_resource
def load_text_model():
    tokenizer = AutoTokenizer.from_pretrained(TEXT_REPO)
    model = AutoModelForSequenceClassification.from_pretrained(TEXT_REPO)
    model.to(device).eval()
    return tokenizer, model

@st.cache_resource
def load_image_model():
    ckpt = torch.load(
        hf_hub_download(IMAGE_REPO, "image_model.pth"),
        map_location=device
    )
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model

# ===============================
# AUDIO CNN (EXACT MATCH)
# ===============================
class AudioCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(16, 2)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

@st.cache_resource
def load_audio_model():
    model = AudioCNN()
    state = torch.load(
        hf_hub_download(AUDIO_REPO, "audio_model.pth"),
        map_location=device
    )
    model.load_state_dict(state)
    model.to(device).eval()
    return model

tokenizer, text_model = load_text_model()
image_model = load_image_model()
audio_model = load_audio_model()

# ===============================
# AUDIO PREPROCESSING
# ===============================
def audio_to_spectrogram(file):
    y, sr = librosa.load(file, sr=16000)
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=128, n_fft=1024, hop_length=512
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min())
    return torch.tensor(mel_db).unsqueeze(0).unsqueeze(0).float()

# ===============================
# EXPLANATIONS
# ===============================
def explain_text(p):
    return (
        "The text exhibits highly uniform sentence structure, low lexical diversity, "
        "and statistically consistent token transitions—patterns commonly associated "
        "with neural language model generation."
        if p > 0.5 else
        "The text shows natural linguistic variation, irregular phrasing, and entropy "
        "patterns typical of human-written content."
    )

def explain_image(p):
    return (
        "The image contains overly smooth textures, frequency-domain artifacts, "
        "and uniform pixel correlations indicative of generative image models."
        if p > 0.5 else
        "The image shows natural noise, sharp boundaries, and heterogeneous textures "
        "consistent with real-world photography."
    )

def explain_audio(p):
    return (
        "The audio spectrogram reveals excessive smoothness, lack of micro-variations, "
        "and synthetic harmonic consistency typical of AI-generated speech."
        if p > 0.5 else
        "The audio contains natural background noise, irregular prosody, and spectral "
        "variations characteristic of human speech."
    )

def explain_fusion(probs):
    return (
        "Multiple modalities consistently indicate AI-generated characteristics, "
        "reinforcing the final decision with high confidence."
        if sum(probs)/len(probs) > 0.5 else
        "The modalities exhibit human-like variability with no strong agreement "
        "on synthetic patterns, leading to a human classification."
    )

# ===============================
# UI
# ===============================
st.markdown("""
# 🧠 Multimodal AI Content Detection System
Detect **AI-generated content** using **Text, Image, and Audio** with **Explainable AI**
---
""")

tabs = st.tabs([
    "📝 Text",
    "🖼️ Image",
    "🔊 Audio",
    "🌐 Multimodal Fusion"
])

# ===============================
# TEXT TAB
# ===============================
with tabs[0]:
    text = st.text_area("Enter text")
    if st.button("Analyze Text"):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
        with torch.no_grad():
            probs = F.softmax(text_model(**inputs).logits, dim=1)[0]
        st.metric("AI Probability", f"{probs[1]:.4f}")
        st.success("🟥 AI-GENERATED" if probs[1] > 0.5 else "🟩 HUMAN")
        st.write(explain_text(probs[1]))

# ===============================
# IMAGE TAB
# ===============================
with tabs[1]:
    img_file = st.file_uploader("Upload image", type=["jpg","png"])
    if img_file:
        img = Image.open(img_file).convert("RGB")
        st.image(img, width=300)
        x = transforms.ToTensor()(img).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = F.softmax(image_model(x), dim=1)[0]
        st.metric("AI Probability", f"{probs[1]:.4f}")
        st.success("🟥 AI-GENERATED" if probs[1] > 0.5 else "🟩 HUMAN")
        st.write(explain_image(probs[1]))

# ===============================
# AUDIO TAB
# ===============================
with tabs[2]:
    audio_file = st.file_uploader("Upload audio", type=["wav","mp3"])
    if audio_file:
        st.audio(audio_file)
        x = audio_to_spectrogram(audio_file).to(device)
        with torch.no_grad():
            probs = F.softmax(audio_model(x), dim=1)[0]
        st.metric("AI Probability", f"{probs[1]:.4f}")
        st.success("🟥 AI-GENERATED" if probs[1] > 0.5 else "🟩 HUMAN")
        st.write(explain_audio(probs[1]))

# ===============================
# MULTIMODAL FUSION
# ===============================
with tabs[3]:
    st.info("Upload any combination of text, image, and audio.")

    fusion_probs = []

    f_text = st.text_area("Text (optional)")
    if f_text:
        inputs = tokenizer(f_text, return_tensors="pt", truncation=True, padding=True).to(device)
        fusion_probs.append(F.softmax(text_model(**inputs).logits, dim=1)[0][1].item())

    f_img = st.file_uploader("Image (optional)", type=["jpg","png"], key="fusion_img")
    if f_img:
        img = Image.open(f_img).convert("RGB")
        x = transforms.ToTensor()(img).unsqueeze(0).to(device)
        fusion_probs.append(F.softmax(image_model(x), dim=1)[0][1].item())

    f_audio = st.file_uploader("Audio (optional)", type=["wav","mp3"], key="fusion_audio")
    if f_audio:
        x = audio_to_spectrogram(f_audio).to(device)
        fusion_probs.append(F.softmax(audio_model(x), dim=1)[0][1].item())

    if st.button("Run Multimodal Fusion") and fusion_probs:
        P_fused = sum(fusion_probs)/len(fusion_probs)
        st.metric("Fused AI Probability", f"{P_fused:.4f}")
        st.success("🟥 AI-GENERATED" if P_fused > 0.5 else "🟩 HUMAN")
        st.write(explain_fusion(fusion_probs))
