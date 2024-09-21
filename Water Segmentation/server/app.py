# app.py
from flask import Flask, request, render_template, redirect, url_for
import os
import numpy as np
import torch
from torchvision import transforms
import json
import tifffile as tiff
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['RESULT_FOLDER'] = 'static/results/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)
mean_std_path = os.path.join('..','mean_std.json')

# model = torch.load('../models_weights/deeplab_finetuned/deeplabv3_resnet50.pth')
model = torch.load('../models_weights/unet_finetuned/unet_plusplus_resnet101.pth')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()
def preprocess_image(image: np.ndarray) -> np.ndarray:
    with open(mean_std_path, 'r') as f:
            mean_std = json.load(f)
    mean = mean_std['mean']
    std = mean_std['std']

    image = np.resize(image, (128, 128, 12))
    image = torch.tensor(image, dtype=torch.float).permute(2, 0, 1)
    image = transforms.Normalize(mean=mean, std=std)(image)
    image = image.unsqueeze(0) 
    image = image.to(device)
    return image

def postprocess_output(pred_mask: torch.Tensor) -> np.ndarray:
    pred_mask = torch.sigmoid(pred_mask) > 0.5
    pred_mask = pred_mask.squeeze().cpu().numpy()
    return pred_mask

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'tif', 'tiff'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        if 'image' not in request.files:
            return 'No image file found in the request', 400
        file = request.files['image']
        if file.filename == '':
            return 'No selected file', 400
        if file and allowed_file(file.filename):
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Process the image
            image = tiff.imread(filepath)
            # view rgb image
            rgb_image = image[:, :, [3, 2, 1]]
            rgb_image = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min()) * 255
            rgb_image = rgb_image.astype('uint8')
            rgb_pil_image = Image.fromarray(rgb_image)
            rgb_filename = 'rgb_' + os.path.splitext(filename)[0] + '.png'
            rgb_filepath = os.path.join(app.config['UPLOAD_FOLDER'], rgb_filename)
            rgb_pil_image.save(rgb_filepath)

            image = preprocess_image(image)
            with torch.no_grad():
                pred_mask = model(image)
            pred_mask = postprocess_output(pred_mask)
            # print("Model output shape:", pred_mask.shape)
            # print("Model output min value:", pred_mask.min().item())
            # print("Model output max value:", pred_mask.max().item())
            mask_image = Image.fromarray((pred_mask * 255).astype('uint8'))
            # Save the mask image
            result_filename = 'mask_' + os.path.splitext(filename)[0] + '.png'
            result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
            mask_image.save(result_path)

            return render_template('result.html',
                                   original_image=url_for('static', filename='uploads/' + rgb_filename),
                                   result_image=url_for('static', filename='results/' + result_filename))
        else:
            return 'Invalid file type. Only .tif and .tiff files are allowed.', 400
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)
