import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

st.set_page_config(page_title="AI Image Captioning", layout="centered")
st.title("🤖 AI Image Caption Generator")
st.write("Upload any image and get an accurate description instantly using BLIP model!")

# 1. Load Pre-trained BLIP Model safely (Cache kar rahe hain taaki fast chale)
@st.cache_resource
def load_blip_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Ye automatic HuggingFace se heavy weights load kar lega
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    return processor, model, device

processor, model, device = load_blip_model()

# 2. UI Layout
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=400)
    
    if st.button("✨ Generate Caption"):
        with st.spinner("AI is analyzing the image..."):
            try:
                # Preprocessing and generation using advanced transformers
                inputs = processor(image, return_tensors="pt").to(device)
                
                with torch.no_grad():
                    out = model.generate(**inputs, max_new_tokens=50)
                    
                # Decode the tokens back to perfect English sentence
                caption = processor.decode(out[0], skip_special_tokens=True)
                
                st.success(f"**Predicted Caption:** {caption.capitalize()}")
                
            except Exception as e:
                st.error("Error generating caption. Please refresh or try again.")