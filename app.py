import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torchvision import models, transforms
from huggingface_hub import hf_hub_download

# ===============================
# STREAMLIT CONFIG
# ===============================
st.set_page_config(
    page_title="Multimodal AI Content Detector",
    layout="wide"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===============================
# HUGGING FACE MODEL REPOS
# ===============================
TEXT_REPO  = "chinmaianjali/text-roberta-ai-detector"
IMAGE_REPO = "chinmaianjali/image-mobilenet-ai-detector"
AUDIO_REPO = "chinmaianjali/audio-cnn-ai-detector"

# ===============================
# LOAD TEXT MODEL
# ===============================
@st.cache_resource
def load_text_model():
    tokenizer = AutoTokenizer.from_pretrained(TEXT_REPO)
    model = AutoModelForSequenceClassification.from_pretrained(TEXT_REPO)
    model.to(device).eval()
    return tokenizer, model

# ===============================
# LOAD IMAGE MODEL
# ===============================
@st.cache_resource
def load_image_model():
    ckpt_path = hf_hub_download(
        repo_id=IMAGE_REPO,
        filename="image_model.pth"
    )

    checkpoint = torch.load(ckpt_path, map_location=device)

    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        checkpoint.get("num_classes", 2)
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model

# ===============================
# LOAD AUDIO MODEL
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
    audio_path = hf_hub_download(
        repo_id=AUDIO_REPO,
        filename="audio_model.pth"
    )
    model = AudioCNN()
    state = torch.load(audio_path, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    return model

# ===============================
# LOAD ALL MODELS (ONCE)
# ===============================
tokenizer, text_model = load_text_model()
image_model = load_image_model()
audio_model = load_audio_model()

# ===============================
# INFERENCE FUNCTIONS
# ===============================
def infer_text_prob(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    ).to(device)
    with torch.no_grad():
        probs = F.softmax(text_model(**inputs).logits, dim=1)
    return probs[0, 1].item()

def infer_image_prob(img):
    tfm = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    x = tfm(img).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = F.softmax(image_model(x), dim=1)
    return probs[0, 1].item()

def infer_audio_prob(x):
    with torch.no_grad():
        probs = F.softmax(audio_model(x.to(device)), dim=1)
    return probs[0, 1].item()

# ===============================
# MULTIMODAL FUSION
# ===============================
def multimodal_fusion(P_text=None, P_audio=None, P_image=None):
    base_w = {"text":0.4, "audio":0.25, "image":0.35}

    def damp(p, w):
        return w * 0.3 if (p < 0.1 or p > 0.9) else w

    probs, weights = [], []

    if P_text is not None:
        probs.append(P_text)
        weights.append(damp(P_text, base_w["text"]))
    if P_audio is not None:
        probs.append(P_audio)
        weights.append(base_w["audio"])
    if P_image is not None:
        probs.append(P_image)
        weights.append(base_w["image"])

    weights = [w / sum(weights) for w in weights]
    P_fused = sum(p * w for p, w in zip(probs, weights))

    label = "AI-GENERATED" if P_fused >= 0.45 else "HUMAN"
    return round(P_fused, 4), label

# ===============================
# STREAMLIT UI
# ===============================
st.title("🧠 Multimodal AI Content Detector")

tabs = st.tabs(["📝 Text", "🖼️ Image", "🌐 Multimodal Fusion"])

# -------------------------------
# TEXT TAB
# -------------------------------
with tabs[0]:
    text = st.text_area("Enter text")
    if st.button("Analyze Text"):
        if text.strip():
            p = infer_text_prob(text)
            st.metric("AI Probability", f"{p:.4f}")
        else:
            st.warning("Please enter some text.")

# -------------------------------
# IMAGE TAB
# -------------------------------
with tabs[1]:
    img_file = st.file_uploader("Upload image", type=["jpg", "png"])
    if img_file:
        img = Image.open(img_file).convert("RGB")
        st.image(img, width=300)
        p = infer_image_prob(img)
        st.metric("AI Probability", f"{p:.4f}")

# -------------------------------
# MULTIMODAL FUSION TAB
# -------------------------------
with tabs[2]:
    p_text  = st.number_input("Text AI Probability", 0.0, 1.0, 0.0)
    p_audio = st.number_input("Audio AI Probability", 0.0, 1.0, 0.0)
    p_image = st.number_input("Image AI Probability", 0.0, 1.0, 0.0)

    if st.button("Run Fusion"):
        p_fused, label = multimodal_fusion(p_text, p_audio, p_image)
        st.metric("Fused AI Probability", f"{p_fused}")
        st.success(f"FINAL DECISION: {label}")
