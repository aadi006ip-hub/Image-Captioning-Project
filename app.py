import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
# Apne local files se import karo (models aur vocab ke mutabik badal lena)
# from src.model import EncoderCNN, DecoderRNN 

st.set_page_config(page_title="Image Captioning Bot", layout="centered")
st.title("🤖 AI Image Caption Generator")
st.write("Upload an image and let the trained LSTM model describe it!")

# 1. Load Model & Vocab (Cache kar rahe hain taaki baar-baar load na ho)
@st.cache_resource
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # NOTE: Yahan apne exact Encoder aur Decoder classes ko initialize karna
    # encoder = EncoderCNN(embed_size=256)
    # decoder = DecoderRNN(embed_size=256, hidden_size=512, vocab_size=len(vocab))
    
    # Weights load karo (path check kar lena)
    # encoder.load_state_dict(torch.load("artifacts/encoder_epoch_15.pth", map_location=device))
    # decoder.load_state_dict(torch.load("artifacts/decoder_epoch_15.pth", map_location=device))
    
    # encoder.eval()
    # decoder.eval()
    
    # return encoder, decoder, device, vocab
    return None, None, device, None

encoder, decoder, device, vocabulary = load_models()

# Transforms Setup
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# 2. UI Layout
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    if st.button("✨ Generate Caption"):
        with st.spinner("Model is thinking..."):
            try:
                # Image preprocessing
                img_tensor = transform(image).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    features = encoder(img_tensor).unsqueeze(1)
                    current_inputs = features
                    result_caption = []
                    
                    for _ in range(20):
                        lstm_out, _ = decoder.lstm(current_inputs)
                        outputs = decoder.linear(lstm_out)
                        
                        next_word_logits = outputs[:, -1, :]
                        predicted = next_word_logits.argmax(1)
                        
                        word = vocabulary.itos[predicted.item()]
                        
                        if word == "<EOS>":
                            break
                        if word != "<SOS>":
                            result_caption.append(word)
                            
                        next_embed = decoder.embed(predicted).unsqueeze(1)
                        current_inputs = torch.cat((current_inputs, next_embed), dim=1)
                
                final_caption = " ".join(result_caption)
                st.success(f"**Predicted Caption:** {final_caption}")
                
            except Exception as e:
                st.error(f"Error generating caption. Please verify token mapping.")