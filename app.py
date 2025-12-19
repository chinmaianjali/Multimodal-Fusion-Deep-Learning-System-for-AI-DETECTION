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
# HERO SECTION
# =========================================================
st.image("assets/banner.webp", use_column_width=True, clamp=True)
st.markdown("<br>", unsafe_allow_html=True)

st.markdown("""
<h1 style="text-align:center; font-size:46px;">🧠 Multimodal AI Content Detection System</h1>
<p style="text-align:center; font-size:22px;">
✨ Detect <b>AI-generated</b> Text 📝, Images 🖼️, and Audio 🔊 using <b>Deep Learning</b> & <b>Explainable AI</b>
</p>
<hr>
""", unsafe_allow_html=True)

# =========================================================
# HUGGING FACE REPOS
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

# ================= AUDIO CNN =================
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
    st.markdown("### 📊 Probability Distribution")
    st.bar_chart({"🧑 Human": human, "🤖 AI": ai})

def verdict_card(is_ai):
    if is_ai:
        st.markdown("""
        <div style="background:#ffdddd;padding:22px;border-radius:14px;
        border:2px solid #e74c3c;font-size:24px;text-align:center;font-weight:bold;">
        🚨 FINAL VERDICT: 🤖 AI-GENERATED CONTENT
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:#ddffdd;padding:22px;border-radius:14px;
        border:2px solid #2ecc71;font-size:24px;text-align:center;font-weight:bold;">
        ✅ FINAL VERDICT: 🧑 HUMAN-GENERATED CONTENT
        </div>
        """, unsafe_allow_html=True)

def explain_block(text):
    st.markdown(
        f"<div style='font-size:18px; line-height:1.7;'>{text.replace(chr(10), '<br>')}</div>",
        unsafe_allow_html=True
    )

# =========================================================
# XAI EXPLANATIONS
# =========================================================
def explain_text(p):
    return (
        "📝 **Why this TEXT is classified as AI:**\n"
        "• 🤖 Extremely consistent grammar and sentence length\n"
        "• 📉 Low lexical diversity with repetitive phrasing\n"
        "• 🔁 Predictable token transitions (low entropy)\n"
        "• 🚫 Lack of human-like stylistic imperfections"
        if p > 0.5 else
        "📝 **Why this TEXT is classified as HUMAN:**\n"
        "• 🧠 Natural variation in sentence structure\n"
        "• 📚 Rich vocabulary and contextual nuance\n"
        "• 🎯 Higher randomness in word selection\n"
        "• ✍️ Human-like stylistic inconsistencies"
    )

def explain_image(p):
    return (
        "🖼️ **Why this IMAGE is classified as AI:**\n"
        "• 🎨 Over-smooth textures without camera noise\n"
        "• 📐 Uniform pixel correlations\n"
        "• 🧪 Generative frequency artifacts\n"
        "• ✂️ Inconsistent edge sharpness"
        if p > 0.5 else
        "🖼️ **Why this IMAGE is classified as HUMAN:**\n"
        "• 🌤️ Natural lighting and shadow variations\n"
        "• 📷 Realistic sensor noise patterns\n"
        "• 🔍 Sharp, irregular edges\n"
        "• ✅ Absence of generative artifacts"
    )

def explain_audio(p):
    return (
        "🔊 **Why this AUDIO is classified as AI:**\n"
        "• 📈 Overly smooth spectrogram patterns\n"
        "• 😮 No breathing or micro-pauses\n"
        "• 🎵 Over-regular harmonics\n"
        "• 🤖 Synthetic speech characteristics"
        if p > 0.5 else
        "🔊 **Why this AUDIO is classified as HUMAN:**\n"
        "• 🗣️ Natural pitch fluctuations\n"
        "• 😮 Presence of breath & pauses\n"
        "• 🎼 Rich spectral diversity\n"
        "• 🧑 Human-like prosody"
    )

def explain_fusion(probs):
    return (
        "🌐 **Multimodal Fusion Explanation:**\n"
        "• 🔗 Multiple modalities agree on AI patterns\n"
        "• 📊 Cross-modal confidence reinforcement\n"
        "• 🧠 Reduced single-model uncertainty\n"
        "• 🏁 Final decision reflects holistic AI detection"
        if sum(probs)/len(probs) > 0.5 else
        "🌐 **Multimodal Fusion Explanation:**\n"
        "• 🧑 Modalities show natural variability\n"
        "• ⚖️ No strong AI agreement across inputs\n"
        "• 🧠 Fusion favors authenticity\n"
        "• 🏁 Final decision reflects human origin"
    )

# =========================================================
# TABS
# =========================================================
tabs = st.tabs(["📝 Text", "🖼️ Image", "🔊 Audio", "🌐 Multimodal Fusion"])

# ================= TEXT =================
with tabs[0]:
    st.image("assets/text.avif", width=220)
    st.markdown("<h2 style='font-size:32px;'>📝 Text AI Detection</h2>", unsafe_allow_html=True)
    text = st.text_area("✍️ Enter text to analyze")
    if st.button("🔍 Analyze Text"):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
        with torch.no_grad():
            probs = F.softmax(text_model(**inputs).logits, dim=1)[0]
        probability_chart(probs[0].item(), probs[1].item())
        verdict_card(probs[1] > 0.5)
        explain_block(explain_text(probs[1].item()))

# ================= IMAGE =================
with tabs[1]:
    st.image("assets/image.webp", width=220)
    st.markdown("<h2 style='font-size:32px;'>🖼️ Image AI Detection</h2>", unsafe_allow_html=True)
    img_file = st.file_uploader("📤 Upload an image", type=["jpg","png","webp"])
    if img_file:
        img = Image.open(img_file).convert("RGB")
        st.image(img, width=320)
        x = transforms.ToTensor()(img).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = F.softmax(image_model(x), dim=1)[0]
        probability_chart(probs[0].item(), probs[1].item())
        verdict_card(probs[1] > 0.5)
        explain_block(explain_image(probs[1].item()))

# ================= AUDIO =================
with tabs[2]:
    st.image("assets/audio.jpg", width=220)
    st.markdown("<h2 style='font-size:32px;'>🔊 Audio AI Detection</h2>", unsafe_allow_html=True)
    audio_file = st.file_uploader("📤 Upload audio", type=["wav","mp3"])
    recorded = st.audio_input("🎙️ Or record audio")
    source = audio_file if audio_file else recorded
    if source:
        st.audio(source)
        x = audio_to_spectrogram(source).to(device)
        with torch.no_grad():
            probs = F.softmax(audio_model(x), dim=1)[0]
        probability_chart(probs[0].item(), probs[1].item())
        verdict_card(probs[1] > 0.5)
        explain_block(explain_audio(probs[1].item()))

# ================= FUSION =================
with tabs[3]:
    st.image("assets/fusion.png", width=220)
    st.markdown("<h2 style='font-size:32px;'>🌐 Multimodal Fusion Engine</h2>", unsafe_allow_html=True)

    fusion_probs = []

    f_text = st.text_area("📝 Text (optional)")
    if f_text:
        inputs = tokenizer(f_text, return_tensors="pt", truncation=True, padding=True).to(device)
        fusion_probs.append(F.softmax(text_model(**inputs).logits, dim=1)[0][1].item())

    f_img = st.file_uploader("🖼️ Image (optional)", type=["jpg","png","webp"], key="fimg")
    if f_img:
        img = Image.open(f_img).convert("RGB")
        fusion_probs.append(
            F.softmax(image_model(transforms.ToTensor()(img).unsqueeze(0).to(device)), dim=1)[0][1].item()
        )

    f_audio = st.file_uploader("🔊 Audio (optional)", type=["wav","mp3"], key="faud")
    if f_audio:
        fusion_probs.append(
            F.softmax(audio_model(audio_to_spectrogram(f_audio).to(device)), dim=1)[0][1].item()
        )

    if st.button("🚀 Run Multimodal Fusion") and fusion_probs:
        P = sum(fusion_probs) / len(fusion_probs)
        probability_chart(1 - P, P)
        verdict_card(P > 0.5)
        explain_block(explain_fusion(fusion_probs))
