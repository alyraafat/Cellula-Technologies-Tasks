import streamlit as st
import torch
from PIL import Image

def deploy(model_weights_path: str, im_target_shape: tuple):
    model = torch.load(model_weights_path)
    transforms = transforms.Compose([
        transforms.Resize(im_target_shape),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    model.eval()
    st.title("Tooth Type Classifier")
    st.write("Upload an image to classify it.")
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")
    if uploaded_file is not None:
        
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)

        # Preprocess the image
        img_tensor = transforms(image).unsqueeze(0)

        # Make prediction
        with torch.no_grad():
            outputs = model(img_tensor)
            _, predicted = torch.argmax(outputs, dim=-1)
        
        # Display the prediction
        st.write(f"Predicted class: {predicted.item()}")

if __name__ == '__main__':
    deploy('./models_weights/best_model.pth', (224, 224))