import io
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import streamlit as st
from PIL import Image

# ==========================================
# 1. CORE MODEL ARCHITECTURE (CNN + LSTM)
# ==========================================
class ImageEncoder(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters(): 
            param.requires_grad = False
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        self.relu = nn.ReLU()

    def forward(self, images):
        return self.relu(self.embed(self.resnet(images).view(images.size(0), -1)))

class CaptionDecoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        embeddings = torch.cat((features.unsqueeze(1), self.embed(captions[:, :-1])), dim=1)
        lstm_out, _ = self.lstm(embeddings)
        return self.linear(lstm_out)

# ==========================================
# 2. VOCABULARY MOCK/LOADER
# ==========================================
# NOTE: Agar tumhare paas vocab pickle/JSON file hai, toh use yahan load kar lena.
# Yeh fallback configuration hai taaki app crash na ho.
class SimpleVocabulary:
    def __init__(self):
        self.itos = {0: "<PAD>", 1: "startseq", 2: "endseq", 3: "a", 4: "man", 5: "is", 6: "climbing", 7: "rock"}
        self.stoi = {v: k for k, v in self.itos.items()}

# ==========================================
# 3. BEAM SEARCH INFERENCE DECODING
# ==========================================
def generate_caption_beam_search(image, encoder, decoder, vocab, device, beam_index=3, max_len=20):
    encoder.eval()
    decoder.eval()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        features = encoder(image_tensor).unsqueeze(1)
    
    start_token = vocab.stoi.get("startseq", 1)
    end_token = vocab.stoi.get("endseq", 2)
    
    sequences = [[[start_token], 0.0]]
    
    for _ in range(max_len):
        all_candidates = []
        for seq, score in sequences:
            if seq[-1] == end_token:
                all_candidates.append([seq, score])
                continue
                
            decoder_input = torch.tensor([seq]).to(device)
            with torch.no_grad():
                # For basic dynamic inference simulation
                embeddings = decoder.embed(decoder_input)
                lstm_out, _ = decoder.lstm(torch.cat((features, embeddings), dim=1))
                log_probs = F.log_softmax(decoder.linear(lstm_out[:, -1, :]), dim=-1)
            
            top_log_probs, top_words = torch.topk(log_probs, beam_index, dim=-1)
            for i in range(beam_index):
                next_word = top_words[0][i].item()
                next_score = top_log_probs[0][i].item()
                all_candidates.append([seq + [next_word], score + next_score])
                
        ordered = sorted(all_candidates, key=lambda x: x[1], reverse=True)
        sequences = ordered[:beam_index]
        if all(seq[0][-1] == end_token for seq in sequences):
            break
            
    best_seq = sequences[0][0]
    caption_words = [vocab.itos.get(idx, "") for idx in best_seq if idx not in [start_token, end_token, 0]]
    return " ".join(caption_words)

# ==========================================
# 4. STREAMLIT USER INTERFACE (UI)
# ==========================================
st.set_page_config(page_title="AI Image Captioner", page_icon="📸", layout="centered")
st.title("📸 AI Image Captioning Engine")
st.write("Upload an image to generate a smart sequential description using CNN-LSTM Architecture.")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocab = SimpleVocabulary()

# Load Weights safely if available
encoder = ImageEncoder(embed_size=256).to(DEVICE)
decoder = CaptionDecoder(embed_size=256, hidden_size=512, vocab_size=len(vocab.itos)).to(DEVICE)

# Path configuration for cloud/local fallback
enc_path = "artifacts/encoder_epoch_5.pth"
dec_path = "artifacts/decoder_epoch_5.pth"

if os.path.exists(enc_path) and os.path.exists(dec_path):
    encoder.load_state_dict(torch.load(enc_path, map_location=DEVICE))
    decoder.load_state_dict(torch.load(dec_path, map_location=DEVICE))
    st.sidebar.success("Loaded model weights from Epoch 5!")
else:
    st.sidebar.warning("Running inference mode with initialized base architecture layers.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Input Image", use_container_width=True)
    
    if st.button("Generate Caption ✨", type="primary"):
        with st.spinner("Processing deep layers & context decoding..."):
            caption = generate_caption_beam_search(image, encoder, decoder, vocab, DEVICE)
            st.success(f"**Predicted Caption:** {caption}")