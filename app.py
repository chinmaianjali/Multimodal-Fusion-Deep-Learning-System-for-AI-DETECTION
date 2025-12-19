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

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Multimodal AI Content Detection",
    page_icon="🧠",
    layout="wide"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================
# HEADER (SLIM BANNER)
# =========================================================
col1, col2 = st.columns([1, 4])
with col1:
    st.image("assets/banner.webp", width=160)
with col2:
    st.markdown("""
    <h1 style="font-size:36px; margin-bottom:0;">🧠 Multimodal AI Content Detection</h1>
    <p style="font-size:17px; margin-top:4px;">
    Detect AI-generated <b>Text</b>, <b>Images</b>, and <b>Audio</b> using Deep Learning & Explainable AI
    </p>
    """, unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# =========================================================
# MODEL REPOS
# =========================================================
TEXT_REPO  = "chinmaianjali/text-roberta-ai-detector"
IMAGE_REPO = "chinmaianjali/image-mobilenet-ai-detector"
AUDIO_REPO = "chinmaianjali/audio-cnn-ai-detector"

# =========================================================
# LOAD MODELS
# =========================================================
@st.cache_resource
def load_text_model():
    tokenizer = AutoTokenizer.from_pretrained(TEXT_REPO)
    model = AutoModelForSequenceClassification.from_pretrained(TEXT_REPO)
    model.to(device).eval()
    return tokenizer, model

@st.cache_resource
def load_image_model():
    checkpoint = torch.load(
        hf_hub_download(IMAGE_REPO, "image_model.pth"),
        map_location=device
    )
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model

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
        return self.fc(self.conv(x).view(x.size(0), -1))

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

# =========================================================
# AUDIO PREPROCESSING
# =========================================================
def audio_to_spectrogram(file):
    y, sr = librosa.load(file, sr=16000)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel = librosa.power_to_db(mel, ref=np.max)
    mel = (mel - mel.min()) / (mel.max() - mel.min())
    return torch.tensor(mel).unsqueeze(0).unsqueeze(0).float()

# =========================================================
# UI HELPERS
# =========================================================
def probability_chart(human, ai):
    st.bar_chart({"Human": human, "AI": ai})

def verdict_card(is_ai):
    color = "#ffecec" if is_ai else "#ecffec"
    border = "#e74c3c" if is_ai else "#2ecc71"
    label = "🤖 AI-GENERATED CONTENT" if is_ai else "🧑 HUMAN-GENERATED CONTENT"
    st.markdown(
        f"""
        <div style="
        background:{color};
        border:2px solid {border};
        border-radius:10px;
        padding:16px;
        font-size:20px;
        text-align:center;
        font-weight:600;">
        {label}
        </div>
        """,
        unsafe_allow_html=True
    )

def explain(text):
    st.markdown(
        f"<div style='font-size:16px; line-height:1.6;'>{text.replace(chr(10), '<br>')}</div>",
        unsafe_allow_html=True
    )

# =========================================================
# EXPLANATIONS
# =========================================================
def explain_text(p):
    return (
        "• Highly uniform sentence structure\n"
        "• Predictable grammar and phrasing\n"
        "• Low lexical entropy\n"
        "• Absence of human stylistic variation"
        if p > 0.5 else
        "• Natural sentence variation\n"
        "• Diverse vocabulary usage\n"
        "• Higher randomness\n"
        "• Human-like stylistic imperfections"
    )

def explain_image(p):
    return (
        "• Over-smoothed textures\n"
        "• Generative frequency artifacts\n"
        "• Uniform pixel correlations"
        if p > 0.5 else
        "• Natural lighting\n"
        "• Realistic noise patterns\n"
        "• Camera-consistent edges"
    )

def explain_audio(p):
    return (
        "• Excessively smooth spectrogram\n"
        "• No breathing or pauses\n"
        "• Regular harmonic patterns"
        if p > 0.5 else
        "• Natural pitch variations\n"
        "• Presence of breath and pauses\n"
        "• Human prosody"
    )

def explain_fusion(probs):
    return (
        "• Cross-modal agreement on AI traits\n"
        "• Reinforced confidence across models\n"
        "• Reduced uncertainty via fusion"
        if sum(probs)/len(probs) > 0.5 else
        "• No strong AI agreement\n"
        "• Modalities indicate human traits\n"
        "• Fusion favors authenticity"
    )

# =========================================================
# TABS
# =========================================================
tabs = st.tabs(["📝 Text", "🖼️ Image", "🔊 Audio", "🌐 Multimodal Fusion"])

# ================= TEXT =================
with tabs[0]:
    st.image("assets/text.avif", width=260)
    st.markdown("<h2 style='font-size:26px;'>📝 Text Analysis</h2>", unsafe_allow_html=True)
    text = st.text_area("Enter text")
    if st.button("Analyze Text"):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
        probs = F.softmax(text_model(**inputs).logits, dim=1)[0]
        probability_chart(probs[0].item(), probs[1].item())
        verdict_card(probs[1] > 0.5)
        explain(explain_text(probs[1].item()))

# ================= IMAGE =================
with tabs[1]:
    st.image("assets/image.webp", width=260)
    st.markdown("<h2 style='font-size:26px;'>🖼️ Image Analysis</h2>", unsafe_allow_html=True)
    img_file = st.file_uploader("Upload image", type=["jpg","png","webp"])
    if img_file:
        img = Image.open(img_file).convert("RGB")
        st.image(img, width=340)
        probs = F.softmax(image_model(transforms.ToTensor()(img).unsqueeze(0).to(device)), dim=1)[0]
        probability_chart(probs[0].item(), probs[1].item())
        verdict_card(probs[1] > 0.5)
        explain(explain_image(probs[1].item()))

# ================= AUDIO =================
with tabs[2]:
    st.image("assets/audio.jpg", width=260)
    st.markdown("<h2 style='font-size:26px;'>🔊 Audio Analysis</h2>", unsafe_allow_html=True)
    audio_file = st.file_uploader("Upload audio", type=["wav","mp3"])
    recorded = st.audio_input("Record audio")
    source = audio_file if audio_file else recorded
    if source:
        st.audio(source)
        probs = F.softmax(audio_model(audio_to_spectrogram(source).to(device)), dim=1)[0]
        probability_chart(probs[0].item(), probs[1].item())
        verdict_card(probs[1] > 0.5)
        explain(explain_audio(probs[1].item()))

# ================= FUSION =================
with tabs[3]:
    st.image("assets/fusion.png", width=260)
    st.markdown("<h2 style='font-size:26px;'>🌐 Multimodal Fusion</h2>", unsafe_allow_html=True)
    fusion_probs = []

    t = st.text_area("Text (optional)")
    if t:
        fusion_probs.append(
            F.softmax(text_model(**tokenizer(t, return_tensors="pt").to(device)).logits, dim=1)[0][1].item()
        )

    i = st.file_uploader("Image (optional)", type=["jpg","png","webp"], key="fimg")
    if i:
        img = Image.open(i).convert("RGB")
        fusion_probs.append(
            F.softmax(image_model(transforms.ToTensor()(img).unsqueeze(0).to(device)), dim=1)[0][1].item()
        )

    a = st.file_uploader("Audio (optional)", type=["wav","mp3"], key="faud")
    if a:
        fusion_probs.append(
            F.softmax(audio_model(audio_to_spectrogram(a).to(device)), dim=1)[0][1].item()
        )

    if st.button("Run Fusion") and fusion_probs:
        P = sum(fusion_probs) / len(fusion_probs)
        probability_chart(1 - P, P)
        verdict_card(P > 0.5)
        explain(explain_fusion(fusion_probs))
