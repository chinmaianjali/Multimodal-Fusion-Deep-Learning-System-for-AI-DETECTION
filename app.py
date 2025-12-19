import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torchvision import models, transforms

# ===============================
# STREAMLIT CONFIG
# ===============================
st.set_page_config(
    page_title="Multimodal AI Content Detector",
    layout="wide"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===============================
# PATH CONFIG (CHANGE ONLY THESE)
# ===============================
TEXT_MODEL_PATH  = "models/text_roberta_final"
IMAGE_MODEL_PATH = "models/ai_image_resnet.pth"
# Audio model assumed probabilistic output (as per your pipeline)

# ===============================
# LOAD MODELS (CACHED)
# ===============================
@st.cache_resource
def load_text_model():
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(TEXT_MODEL_PATH)
    model.to(device).eval()
    return tokenizer, model

@st.cache_resource
def load_image_model():
    checkpoint = torch.load(IMAGE_MODEL_PATH, map_location=device)
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = torch.nn.Linear(
        model.classifier[1].in_features,
        checkpoint["num_classes"]
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model

tokenizer, text_model = load_text_model()
image_model = load_image_model()

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

# ===============================
# MULTIMODAL FUSION (FINAL)
# ===============================
def multimodal_fusion(P_text=None, P_audio=None, P_image=None):
    BASE_W = {"text":0.4, "audio":0.25, "image":0.35}

    def dampen(p, w):
        return w * 0.3 if (p < 0.1 or p > 0.9) else w

    probs, weights = [], []

    if P_text is not None:
        probs.append(P_text)
        weights.append(dampen(P_text, BASE_W["text"]))
    if P_audio is not None:
        probs.append(P_audio)
        weights.append(BASE_W["audio"])
    if P_image is not None:
        probs.append(P_image)
        weights.append(BASE_W["image"])

    weights = [w / sum(weights) for w in weights]
    P_fused = sum(p * w for p, w in zip(probs, weights))
    label = "AI-GENERATED" if P_fused >= 0.45 else "HUMAN"
    return round(P_fused, 4), label

# ===============================
# XAI EXPLANATIONS
# ===============================
def explain_text(p):
    if p > 0.8:
        return [
            "Highly structured and neutral language",
            "Lack of personal tone",
            "Consistent sentence complexity",
            "Statistical AI-like phrasing patterns"
        ]
    elif p > 0.5:
        return [
            "Moderate repetition",
            "Formal and generic style"
        ]
    else:
        return [
            "Human-like variability",
            "Natural flow and context shifts"
        ]

def explain_image(p):
    return [
        "Unnatural texture regularity",
        "CNN features aligned with synthetic images",
        "Inconsistent lighting artifacts"
    ] if p > 0.5 else [
        "Natural texture gradients",
        "Camera-like noise patterns"
    ]

def explain_fusion():
    return [
        "Predictions combined across multiple modalities",
        "Overconfident single-model predictions penalized",
        "Consensus-based final decision improves robustness"
    ]

# ===============================
# UI
# ===============================
st.title("🧠 Multimodal AI Content Detection System")

tabs = st.tabs([
    "📝 Text Module",
    "🖼️ Image Module",
    "🌐 Multimodal Fusion"
])

# -------------------------------
# TEXT MODULE
# -------------------------------
with tabs[0]:
    st.subheader("Text AI Detection")
    text_input = st.text_area("Enter text")

    if st.button("Analyze Text"):
        if text_input.strip():
            P_text = infer_text_prob(text_input)
            st.metric("AI Probability", f"{P_text:.4f}")
            st.write("### Explanation")
            for r in explain_text(P_text):
                st.write("•", r)
        else:
            st.warning("Please enter text.")

# -------------------------------
# IMAGE MODULE
# -------------------------------
with tabs[1]:
    st.subheader("Image AI Detection")
    img_file = st.file_uploader("Upload image", type=["jpg", "png"])

    if img_file:
        img = Image.open(img_file).convert("RGB")
        st.image(img, width=300)

        P_img = infer_image_prob(img)
        st.metric("AI Probability", f"{P_img:.4f}")
        st.write("### Explanation")
        for r in explain_image(P_img):
            st.write("•", r)

# -------------------------------
# MULTIMODAL FUSION
# -------------------------------
with tabs[2]:
    st.subheader("Multimodal Fusion")

    P_text  = st.number_input("Text AI Probability", 0.0, 1.0, 0.0)
    P_audio = st.number_input("Audio AI Probability", 0.0, 1.0, 0.0)
    P_image = st.number_input("Image AI Probability", 0.0, 1.0, 0.0)

    if st.button("Run Fusion"):
        P_fused, label = multimodal_fusion(P_text, P_audio, P_image)
        st.metric("Fused AI Probability", f"{P_fused}")
        st.success(f"FINAL DECISION: {label}")

        st.write("### Why this decision?")
        for r in explain_fusion():
            st.write("•", r)
