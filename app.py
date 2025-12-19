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
    page_title="Multimodal AI Detection",
    page_icon="🧠",
    layout="wide"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================
# HEADER
# =========================================================
h1, h2 = st.columns([2, 4])
with h1:
    st.image("assets/banner.webp", width=350)
with h2:
    st.markdown("""
    <h1 style="font-size:36px; margin-bottom:4px;">🧠 Multimodal AI Content Detection</h1>
    <p style="font-size:17px;">
    ✨ Detect 🤖 AI-generated 📝 Text, 🖼️ Images, and 🔊 Audio using Deep Learning & Explainable AI
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
    tok = AutoTokenizer.from_pretrained(TEXT_REPO)
    mdl = AutoModelForSequenceClassification.from_pretrained(TEXT_REPO)
    mdl.to(device).eval()
    return tok, mdl

@st.cache_resource
def load_image_model():
    ckpt = torch.load(hf_hub_download(IMAGE_REPO, "image_model.pth"), map_location=device)
    mdl = models.mobilenet_v2(weights=None)
    mdl.classifier[1] = nn.Linear(mdl.classifier[1].in_features, 2)
    mdl.load_state_dict(ckpt["model_state_dict"])
    mdl.to(device).eval()
    return mdl

class AudioCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1,16,3,padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Dropout(0.3), nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Dropout(0.3), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.fc = nn.Sequential(
            nn.Linear(32,16), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(16,2)
        )
    def forward(self,x):
        return self.fc(self.conv(x).view(x.size(0),-1))

@st.cache_resource
def load_audio_model():
    mdl = AudioCNN()
    state = torch.load(hf_hub_download(AUDIO_REPO, "audio_model.pth"), map_location=device)
    mdl.load_state_dict(state)
    mdl.to(device).eval()
    return mdl

tokenizer, text_model = load_text_model()
image_model = load_image_model()
audio_model = load_audio_model()

# =========================================================
# AUDIO PREPROCESS
# =========================================================
def audio_to_spec(file):
    y, sr = librosa.load(file, sr=16000)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel = librosa.power_to_db(mel, ref=np.max)
    mel = (mel - mel.min()) / (mel.max() - mel.min())
    return torch.tensor(mel).unsqueeze(0).unsqueeze(0).float()

# =========================================================
# UI HELPERS
# =========================================================
def bar_probs(h, a):
    st.bar_chart({"🧑 Human": h, "🤖 AI": a})

def verdict(ai):
    st.markdown(
        f"""
        <div style="
        background:{'#ffecec' if ai else '#ecffec'};
        border:2px solid {'#e74c3c' if ai else '#2ecc71'};
        padding:16px;border-radius:10px;
        font-size:20px;text-align:center;font-weight:600;">
        {'🚨 🤖 AI-GENERATED CONTENT' if ai else '✅ 🧑 HUMAN-GENERATED CONTENT'}
        </div>
        """,
        unsafe_allow_html=True
    )

def explain(txt):
    st.markdown(
        f"<div style='font-size:16px; line-height:1.6;'>{txt.replace(chr(10),'<br>')}</div>",
        unsafe_allow_html=True
    )

# =========================================================
# EXPLANATIONS
# =========================================================
def exp_text(p):
    return (
        "📝 **AI Indicators:**\n"
        "• 🤖 Uniform sentence length\n"
        "• 🔁 Repetitive phrasing\n"
        "• 📉 Low linguistic entropy\n"
        "• 🚫 Missing human irregularities"
        if p>0.5 else
        "📝 **Human Indicators:**\n"
        "• 🧠 Natural variation\n"
        "• 📚 Rich vocabulary\n"
        "• 🎯 Higher randomness\n"
        "• ✍️ Human imperfections"
    )

def exp_image(p):
    return (
        "🖼️ **AI Indicators:**\n"
        "• 🎨 Over-smooth textures\n"
        "• 📐 Uniform pixels\n"
        "• 🧪 GAN artifacts"
        if p>0.5 else
        "🖼️ **Human Indicators:**\n"
        "• 📷 Sensor noise\n"
        "• 🌤️ Natural lighting\n"
        "• 🔍 Sharp edges"
    )

def exp_audio(p):
    return (
        "🔊 **AI Indicators:**\n"
        "• 📈 Smooth spectrograms\n"
        "• 😮 No breaths\n"
        "• 🎵 Regular harmonics"
        if p>0.5 else
        "🔊 **Human Indicators:**\n"
        "• 🗣️ Pitch variation\n"
        "• 😮 Breathing\n"
        "• 🎼 Natural prosody"
    )

def exp_fusion(ps):
    return (
        "🌐 **Fusion Insight:**\n"
        "• 🔗 Cross-modal AI agreement\n"
        "• 📊 Reinforced confidence\n"
        "• 🧠 Reduced uncertainty"
        if sum(ps)/len(ps)>0.5 else
        "🌐 **Fusion Insight:**\n"
        "• 🧑 Human consistency\n"
        "• ⚖️ Weak AI agreement\n"
        "• 🧠 Authentic signal"
    )

# =========================================================
# TABS
# =========================================================
tabs = st.tabs(["📝 Text", "🖼️ Image", "🔊 Audio", "🌐 Fusion"])

# ================= TEXT =================
with tabs[0]:
    c1,c2 = st.columns([2,5])
    with c1: st.image("assets/text.avif", width=300)
    with c2:
        st.markdown("<h2 style='font-size:26px;'>📝 Text Analysis</h2>", unsafe_allow_html=True)
        t = st.text_area("✍️ Enter text")
        if st.button("🔍 Analyze Text"):
            p = F.softmax(text_model(**tokenizer(t, return_tensors="pt").to(device)).logits, dim=1)[0]
            bar_probs(p[0].item(), p[1].item())
            verdict(p[1]>0.5)
            explain(exp_text(p[1]))

# ================= IMAGE =================
with tabs[1]:
    c1,c2 = st.columns([2,5])
    with c1: st.image("assets/image.webp", width=300)
    with c2:
        st.markdown("<h2 style='font-size:26px;'>🖼️ Image Analysis</h2>", unsafe_allow_html=True)
        img = st.file_uploader("📤 Upload image", type=["jpg","png","webp"])
        if img:
            im = Image.open(img).convert("RGB")
            p = F.softmax(image_model(transforms.ToTensor()(im).unsqueeze(0).to(device)), dim=1)[0]
            bar_probs(p[0].item(), p[1].item())
            verdict(p[1]>0.5)
            explain(exp_image(p[1]))

# ================= AUDIO =================
with tabs[2]:
    c1,c2 = st.columns([2,5])
    with c1: st.image("assets/audio.jpg", width=300)
    with c2:
        st.markdown("<h2 style='font-size:26px;'>🔊 Audio Analysis</h2>", unsafe_allow_html=True)
        a = st.file_uploader("📤 Upload audio", type=["wav","mp3"])
        
        src = a 
        if src:
            p = F.softmax(audio_model(audio_to_spec(src).to(device)), dim=1)[0]
            bar_probs(p[0].item(), p[1].item())
            verdict(p[1]>0.5)
            explain(exp_audio(p[1]))

# ================= FUSION =================
with tabs[3]:
    c1,c2 = st.columns([2,5])
    with c1: st.image("assets/fusion.png", width=300)
    with c2:
        st.markdown("<h2 style='font-size:26px;'>🌐 Multimodal Fusion</h2>", unsafe_allow_html=True)
        ps=[]
        t = st.text_area("📝 Text (optional)")
        if t: ps.append(F.softmax(text_model(**tokenizer(t, return_tensors="pt").to(device)).logits, dim=1)[0][1].item())
        i = st.file_uploader("🖼️ Image (optional)", type=["jpg","png","webp"], key="fi")
        if i: ps.append(F.softmax(image_model(transforms.ToTensor()(Image.open(i).convert("RGB")).unsqueeze(0).to(device)), dim=1)[0][1].item())
        a = st.file_uploader("🔊 Audio (optional)", type=["wav","mp3"], key="fa")
        if a: ps.append(F.softmax(audio_model(audio_to_spec(a).to(device)), dim=1)[0][1].item())
        if st.button("🚀 Run Fusion") and ps:
            P=sum(ps)/len(ps)
            bar_probs(1-P,P)
            verdict(P>0.5)
            explain(exp_fusion(ps))
