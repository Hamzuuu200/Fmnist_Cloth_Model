import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import numpy as np
import plotly.express as px

# Set background image
st.markdown("""
<style>
.stApp {
  background-image: url('YOUR_IMAGE_URL_HERE');
  background-size: cover;
  background-position: center;
  background-repeat: no-repeat;
  background-attachment: fixed;
}
.stApp::before {
  content: "";
  position: fixed;
  top: 0; left: 0; width: 100%; height: 100%;
  background-color: rgba(0, 0, 0, 0.3);  /* optional overlay to make text readable */
  z-index: -1;
}
</style>
""", unsafe_allow_html=True)


# --- Background Image ---
page_bg_img = """
<style>
.stApp {
background-image: url("https://i.postimg.cc/v8kqL8my/black-trianglify.jpg");
background-size: cover;
background-position: center;
background-repeat: no-repeat;
background-attachment: fixed;
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# --- Load Model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vgg16 = models.vgg16(pretrained=True)
for param in vgg16.features.parameters():
    param.requires_grad = False

vgg16.classifier = nn.Sequential(
    nn.Linear(25088 , 1024),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(1024 , 512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512 , 10)
)

vgg16.load_state_dict(torch.load("vgg16_weights.pth", map_location=device))
vgg16 = vgg16.to(device)
vgg16.eval()

# --- Preprocessing ---
custom_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --- Class Names ---
class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"
]

# --- Streamlit Layout ---
st.set_page_config(page_title="Fashion MNIST Classifier", layout="wide")

# --- Sidebar with colored cards ---
st.sidebar.markdown("""
<div style="background: linear-gradient(135deg, #ff416c, #ff4b2b);color:white;padding:15px;border-radius:10px;margin-bottom:10px">
<h3>Instructions</h3>
<ul>
<li>Upload an image of a fashion item</li>
<li>Model predicts its class and confidence</li>
<li>Toggle probability chart below</li>
</ul>
</div>

<div style="background: linear-gradient(135deg, #1e3c72, #2a5298);color:white;padding:15px;border-radius:10px;margin-bottom:10px">
<h3>Options</h3>
<ul>
<li>Show probabilities chart</li>
<li>Hover over bars to see exact confidence</li>
</ul>
</div>

<div style="background: linear-gradient(135deg, #11998e, #38ef7d);color:white;padding:15px;border-radius:10px;margin-bottom:10px">
<h3>About</h3>
<p>This app uses a VGG16 model trained on the Fashion MNIST dataset.</p>
</div>
""", unsafe_allow_html=True)

show_probs = st.sidebar.checkbox("Show Class Probabilities", value=True)

# --- Page Title ---
st.title("👗 Fashion MNIST Classifier")
st.markdown("Upload an image and the model will predict its class. See probabilities if enabled.")

# --- File uploader ---
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    # Columns: Image and Predictions
    col1, col2 = st.columns([1, 1.5])

    with col1:
        st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img_tensor = custom_transform(image).unsqueeze(0).to(device)

    # Prediction
    with torch.no_grad():
        outputs = vgg16(img_tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        predicted_index = np.argmax(probs)
        predicted_class = class_names[predicted_index]
        confidence = probs[predicted_index]

    with col2:
        st.markdown(f"### Predicted Class: **{predicted_class}**")
        st.markdown(f"**Confidence:** {confidence*100:.2f}%")

        if show_probs:
            # --- Original Tinted Plotly Bar Chart ---
            df_probs = {
                "Class": class_names,
                "Probability": probs
            }
            fig = px.bar(
                df_probs,
                x="Probability",
                y="Class",
                orientation="h",
                text=np.round(probs, 2),
                color="Probability",
                color_continuous_scale="sunset"
            )
            fig.update_layout(
                xaxis_title="Probability",
                yaxis_title="Class",
                yaxis=dict(autorange="reversed"),
                margin=dict(l=20, r=20, t=20, b=20),
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
