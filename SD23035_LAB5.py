import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import json
import urllib.request

# Step 1: Streamlit page configuration
st.set_page_config(page_title="Image Classification with ResNet18", layout="wide")
st.title("CPU-Based Image Classification using ResNet18")

# Step 3: Force CPU usage
device = torch.device("cpu")

# Step 4: Load pre-trained ResNet18 model
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.eval()
model.to(device)

# Step 5: Image preprocessing
weights = models.ResNet18_Weights.DEFAULT
transform = weights.transforms()

# Load ImageNet class labels
url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
class_labels = urllib.request.urlopen(url).read().decode("utf-8").splitlines()

# Step 6: Image upload interface
uploaded_file = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Step 7: Convert image to tensor
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Model inference
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

    # Step 8: Top-5 predictions
    top5_prob, top5_catid = torch.topk(probabilities, 5)

    results = []
    for i in range(5):
        results.append({
            "Class": class_labels[top5_catid[i]],
            "Probability (%)": round(top5_prob[i].item() * 100, 2)
        })

    df = pd.DataFrame(results)
    st.subheader("Top-5 Predictions")
    st.table(df)

    # Step 9: Bar chart visualization
    st.subheader("Prediction Probability Distribution")
    fig, ax = plt.subplots()
    ax.barh(df["Class"], df["Probability (%)"])
    ax.set_xlabel("Probability (%)")
    ax.set_title("Top-5 Prediction Confidence")
    st.pyplot(fig)
