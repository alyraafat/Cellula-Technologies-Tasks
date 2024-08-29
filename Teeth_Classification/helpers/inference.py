import torch
import matplotlib.pyplot as plt
import random

def visualize_predictions(model, test_dataset, device=None, num_samples=5):
    """
    Visualize predictions on random samples from the test_dataset.

    Parameters:
    - model: The trained model used for inference.
    - test_dataset: The dataset from which random samples will be drawn.
    - test_loader: The DataLoader for the test dataset.
    - device: The device to run the inference on ('cpu' or 'cuda').
    - num_samples: The number of random samples to visualize.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)
    model.eval()

    # Choose random indices for the number of samples requested
    indices = random.sample(range(len(test_dataset)), num_samples)
    correct = 0
    # Iterate over the selected indices
    for i in indices:
        # Get the image and label from the dataset
        image, true_label = test_dataset[i]

        # Unsqueeze to add batch dimension and move to device
        image = image.unsqueeze(0).to(device)

        # Run inference
        with torch.no_grad():
            output = model(image)
            predicted_label = torch.argmax(output, dim=1).item()

        # Move image back to CPU and remove batch dimension
        image = image.squeeze(0).cpu()

        # Plot the image
        plt.imshow(image.permute(1, 2, 0))

        # Set the color of the predicted label
        correct += predicted_label == true_label
        label_color = 'green' if predicted_label == true_label else 'red'

        plt.title(f'True: {true_label}, Pred: {predicted_label}', color=label_color)
        plt.axis('off')
        plt.show()
    print(f'Correct: {correct}, incorrect: {num_samples-correct}')