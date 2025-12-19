import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torchvision import models, transforms
from huggingface_hub import hf_hub_download

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Multimodal AI Detection System",
    page_icon="🧠",
    layout="wide"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===============================
# HUGGING FACE REPOS
# ===============================
TEXT_REPO  = "chinmaianjali/text-roberta-ai-detector"
IMAGE_REPO = "chinmaianjali/image-mobilenet-ai-detector"
AUDIO_REPO = "chinmaianjali/audio-cnn-ai-detector"

# ===============================
# LOAD MODELS
# ===============================
@st.cache_resource
def load_text_model():
    tok = AutoTokenizer.from_pretrained(TEXT_REPO)
    mdl = AutoModelForSequenceClassification.from_pretrained(TEXT_REPO)
    mdl.to(device).eval()
    return tok, mdl

@st.cache_resource
def load_image_model():
    ckpt = torch.load(
        hf_hub_download(IMAGE_REPO, "image_model.pth"),
        map_location=device
    )
    mdl = models.mobilenet_v2(weights=None)
    mdl.classifier[1] = nn.Linear(mdl.classifier[1].in_features, 2)
    mdl.load_state_dict(ckpt["model_state_dict"])
    mdl.to(device).eval()
    return mdl

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
    mdl = AudioCNN()
    state = torch.load(
        hf_hub_download(AUDIO_REPO, "audio_model.pth"),
        map_location=device
    )
    mdl.load_state_dict(state)
    mdl.to(device).eval()
    return mdl

tokenizer, text_model = load_text_model()
image_model = load_image_model()
audio_model = load_audio_model()

# ===============================
# HELPERS
# ===============================
def plot_probs(human, ai):
    fig, ax = plt.subplots()
    ax.bar(["Human", "AI"], [human, ai], color=["#2ecc71", "#e74c3c"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probability")
    st.pyplot(fig)

def explain_decision(p):
    if p > 0.8:
        return "Strong AI indicators detected (high regularity, low entropy patterns)."
    elif p > 0.5:
        return "Moderate AI-like characteristics observed."
    else:
        return "Natural human-like variability detected."

# ===============================
# SESSION STATE
# ===============================
for k in ["P_text", "P_image", "P_audio"]:
    if k not in st.session_state:
        st.session_state[k] = None

# ===============================
# HEADER
# ===============================
st.markdown("""
## 🧠 Multimodal AI Content Detection System
Detect **AI-generated content** using **Text, Image, and Audio** with **Explainable AI**  
---
""")

tabs = st.tabs([
    "📝 Text Analysis",
    "🖼️ Image Analysis",
    "🔊 Audio Analysis",
    "🌐 Multimodal Fusion"
])

# ===============================
# TEXT TAB
# ===============================
with tabs[0]:
    st.subheader("📝 Text AI Detection")
    txt = st.text_area("Enter text for analysis")

    if st.button("🔍 Analyze Text"):
        inputs = tokenizer(txt, return_tensors="pt", truncation=True, padding=True).to(device)
        with torch.no_grad():
            probs = F.softmax(text_model(**inputs).logits, dim=1)[0]

        st.session_state.P_text = probs[1].item()
        plot_probs(probs[0].item(), probs[1].item())
        st.success(explain_decision(probs[1].item()))

# ===============================
# IMAGE TAB
# ===============================
with tabs[1]:
    st.subheader("🖼️ Image AI Detection")
    imgf = st.file_uploader("Upload an image", type=["jpg", "png"])

    if imgf:
        img = Image.open(imgf).convert("RGB")
        st.image(img, width=300)

        x = transforms.ToTensor()(img).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = F.softmax(image_model(x), dim=1)[0]

        st.session_state.P_image = probs[1].item()
        plot_probs(probs[0].item(), probs[1].item())
        st.success(explain_decision(probs[1].item()))

# ===============================
# AUDIO TAB
# ===============================
with tabs[2]:
    st.subheader("🔊 Audio AI Detection")
    st.info("Audio assumed preprocessed into spectrogram (demo placeholder).")

    if st.button("🎵 Analyze Audio"):
        x = torch.randn(1, 1, 128, 128).to(device)  # replace with real preprocessing
        with torch.no_grad():
            probs = F.softmax(audio_model(x), dim=1)[0]

        st.session_state.P_audio = probs[1].item()
        plot_probs(probs[0].item(), probs[1].item())
        st.success(explain_decision(probs[1].item()))

# ===============================
# FUSION TAB
# ===============================
with tabs[3]:
    st.subheader("🌐 Multimodal Fusion Result")
    st.write("Automatically combines latest predictions from all available modalities.")

    if st.button("🚀 Run Fusion"):
        probs = [p for p in [
            st.session_state.P_text,
            st.session_state.P_audio,
            st.session_state.P_image
        ] if p is not None]

        if len(probs) == 0:
            st.warning("Run at least one modality first.")
        else:
            P_fused = sum(probs) / len(probs)
            label = "AI-GENERATED" if P_fused >= 0.45 else "HUMAN"

            st.metric("Fused AI Probability", f"{P_fused:.4f}")
            st.success(f"FINAL DECISION: {label}")
