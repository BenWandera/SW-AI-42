"""
Streamlit App for EcoWaste AI - Waste Classification
Deployed on Streamlit Cloud with real MobileViT model
"""

import streamlit as st
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import MobileViTImageProcessor
import sys
import os

# Add api folder to path
api_path = os.path.join(os.path.dirname(__file__), 'api')
if os.path.exists(api_path):
    sys.path.insert(0, api_path)

# Import model loader
try:
    from model_loader import load_trained_model
except ImportError:
    from api.model_loader import load_trained_model

st.set_page_config(
    page_title="EcoWaste AI - Waste Classifier",
    page_icon="♻️",
    layout="wide"
)

# Class names
CLASS_NAMES = [
    'Cardboard', 'Food Organics', 'Glass', 'Metal',
    'Miscellaneous Trash', 'Paper', 'Plastic', 'Textile Trash', 'Vegetation'
]

CLASS_TO_CATEGORY = {
    'Cardboard': {'id': 'paper', 'name': 'Paper & Cardboard', 'points': 12},
    'Food Organics': {'id': 'organic', 'name': 'Organic/Food Waste', 'points': 10},
    'Glass': {'id': 'glass', 'name': 'Glass', 'points': 18},
    'Metal': {'id': 'metal', 'name': 'Metal', 'points': 20},
    'Miscellaneous Trash': {'id': 'miscellaneous', 'name': 'Miscellaneous', 'points': 5},
    'Paper': {'id': 'paper', 'name': 'Paper & Cardboard', 'points': 12},
    'Plastic': {'id': 'plastic', 'name': 'Plastic', 'points': 15},
    'Textile Trash': {'id': 'miscellaneous', 'name': 'Miscellaneous', 'points': 5},
    'Vegetation': {'id': 'vegetation', 'name': 'Vegetation', 'points': 8}
}

@st.cache_resource
def load_model():
    """Load the model (cached)"""
    try:
        device = torch.device("cpu")  # Streamlit uses CPU
        model, classes = load_trained_model("best_mobilevit_waste_model.pth", device=device)
        processor = MobileViTImageProcessor.from_pretrained("apple/mobilevit-small")
        return model, processor, device, classes
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None, None

def classify_image(image, model, processor, device):
    """Classify the image"""
    try:
        # Preprocess
        inputs = processor(images=image, return_tensors="pt")
        pixel_values = inputs['pixel_values'].to(device)
        
        # Inference
        with torch.no_grad():
            logits = model(pixel_values)
            probs = F.softmax(logits, dim=-1)
            probs_numpy = probs.cpu().numpy()[0]
            
            predicted_idx = torch.argmax(logits, dim=-1).item()
            confidence = float(probs_numpy[predicted_idx])
            predicted_class = CLASS_NAMES[predicted_idx]
        
        return predicted_class, confidence, probs_numpy
    except Exception as e:
        st.error(f"Classification error: {e}")
        return None, None, None

# UI
st.title("♻️ EcoWaste AI - Waste Classification")
st.markdown("Upload a waste item image to classify it using our trained MobileViT model")

# Load model
with st.spinner("Loading model..."):
    model, processor, device, classes = load_model()

if model is None:
    st.error("Failed to load model. Please check if best_mobilevit_waste_model.pth exists.")
else:
    st.success("✅ Model loaded successfully!")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        # Display image
        image = Image.open(uploaded_file).convert('RGB')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption='Uploaded Image', use_container_width=True)
        
        with col2:
            # Classify
            with st.spinner("Classifying..."):
                predicted_class, confidence, probs = classify_image(image, model, processor, device)
            
            if predicted_class:
                category_info = CLASS_TO_CATEGORY[predicted_class]
                
                st.markdown("### Classification Result")
                st.metric("Category", category_info['name'])
                st.metric("Confidence", f"{confidence*100:.2f}%")
                st.metric("Points Earned", category_info['points'])
                
                # Show all predictions
                st.markdown("### All Predictions")
                for i, class_name in enumerate(CLASS_NAMES):
                    prob = probs[i]
                    st.progress(float(prob), text=f"{class_name}: {prob*100:.1f}%")

# Footer
st.markdown("---")
st.markdown("**EcoWaste AI** - Powered by MobileViT + GNN | Built with ❤️ for a cleaner planet")
