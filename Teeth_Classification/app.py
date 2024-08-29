import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import os

def deploy(model_weights_path: str, im_target_shape: tuple):
    class_names = ['CaS', 'CoS', 'Gum', 'MC', 'OC', 'OLP', 'OT']
    model = torch.load(model_weights_path, map_location=torch.device('cpu'), weights_only=False)
    transform = transforms.Compose([
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
        img_tensor = transform(image).unsqueeze(0)

        # Make prediction
        with torch.no_grad():
            outputs = model(img_tensor)
            predicted = torch.argmax(outputs, dim=-1)
        
        # Display the prediction
        pred_class = class_names[predicted.item()]
        st.write(f"Predicted class: {pred_class}")

if __name__ == '__main__':
    model_path = os.path.join('models_weights', 'resnet18', 'best_model.pth')
    deploy(model_path, (224, 224))