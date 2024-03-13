import os

import numpy as np
import timm
import torch
from matplotlib import pyplot as plt
from torchvision import transforms as T
from PIL import Image

from src.dataset_handler import DatasetHandler


def run():
    # Set up paths and directories
    base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
    data_dir = os.path.join(base_dir, "data")
    model_path = os.path.join(base_dir, "models")

    # Set up device and transformations
    device, im_size = "cpu", 224
    transformations = T.Compose([T.Resize((im_size, im_size)), T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    # Load class names and the models
    _, _, _, color_classes = DatasetHandler.create_data_loaders(root=data_dir, transformations=transformations, batch_size=32)
    _, _, _, category_classes = DatasetHandler.create_data_loaders(root=data_dir, transformations=transformations, batch_size=32)

    color_model_path = os.path.join(model_path, 'ecommerce_best_model_color.pth')
    category_model_path = os.path.join(model_path, 'ecommerce_best_model_subcat1.pth')

    color_model = timm.create_model("rexnet_150", pretrained=True, num_classes=47)
    color_model.load_state_dict(torch.load(color_model_path, map_location=device))
    color_model.to(device).eval()

    category_model = timm.create_model("rexnet_150", pretrained=True, num_classes=45)
    category_model.load_state_dict(torch.load(category_model_path, map_location=device))
    category_model.to(device).eval()

    # Load and preprocess the new image
    new_image_path = os.path.join(base_dir, "input", "women-cargo-pants.jpg")
    new_image = Image.open(new_image_path).convert("RGB")
    input_data = transformations(new_image).unsqueeze(0).to(device)

    # Perform inference for color prediction
    with torch.no_grad():
        color_predictions = color_model(input_data)

    color_probs = torch.nn.functional.softmax(color_predictions, dim=1)
    predicted_color_index = torch.argmax(color_probs, dim=1).item()
    predicted_color_name = list(color_classes.keys())[predicted_color_index]

    # Perform inference for category prediction
    with torch.no_grad():
        category_predictions = category_model(input_data)

    category_probs = torch.nn.functional.softmax(category_predictions, dim=1)
    predicted_category_index = torch.argmax(category_probs, dim=1).item()
    predicted_category_name = list(category_classes.keys())[predicted_category_index]

    # Display results
    print(f"Predicted Color: {predicted_color_name}")
    print(f"Predicted Category: {predicted_category_name}")

    response = {
        "predicted_color": predicted_color_name,
        "predicted_category": predicted_category_name
    }
    print(response)
    return response

    plt.imshow(np.array(new_image))
    plt.title(f"Predicted Color: {predicted_color_name}\nPredicted Category: {predicted_category_name}")
    plt.axis('off')
    plt.show()

def get_recommendations(image):
    base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
    data_dir = os.path.join(base_dir, "data")
    model_path = os.path.join(base_dir, "models")

    # Set up device and transformations
    device, im_size = "cpu", 224
    transformations = T.Compose([T.Resize((im_size, im_size)), T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    # Load class names and the models
    _, _, _, color_classes = DatasetHandler.create_data_loaders(root=data_dir, transformations=transformations, batch_size=32)
    _, _, _, category_classes = DatasetHandler.create_data_loaders(root=data_dir, transformations=transformations, batch_size=32)

    color_model_path = os.path.join(model_path, 'ecommerce_best_model_color.pth')
    category_model_path = os.path.join(model_path, 'ecommerce_best_model_subcat1.pth')

    color_model = timm.create_model("rexnet_150", pretrained=True, num_classes=47)
    color_model.load_state_dict(torch.load(color_model_path, map_location=device))
    color_model.to(device).eval()

    category_model = timm.create_model("rexnet_150", pretrained=True, num_classes=45)
    category_model.load_state_dict(torch.load(category_model_path, map_location=device))
    category_model.to(device).eval()

    # Load and preprocess the new image
    new_image = Image.open(image).convert("RGB")
    input_data = transformations(new_image).unsqueeze(0).to(device)

    # Perform inference for color prediction
    with torch.no_grad():
        color_predictions = color_model(input_data)

    color_probs = torch.nn.functional.softmax(color_predictions, dim=1)
    predicted_color_index = torch.argmax(color_probs, dim=1).item()
    predicted_color_name = list(color_classes.keys())[predicted_color_index]

    # Perform inference for category prediction
    with torch.no_grad():
        category_predictions = category_model(input_data)

    category_probs = torch.nn.functional.softmax(category_predictions, dim=1)
    predicted_category_index = torch.argmax(category_probs, dim=1).item()
    predicted_category_name = list(category_classes.keys())[predicted_category_index]

    # Display results
    print(f"Predicted Color: {predicted_color_name}")
    print(f"Predicted Category: {predicted_category_name}")

    response = {
        "predicted_color": predicted_color_name,
        "predicted_category": predicted_category_name
    }

    return response


if __name__ == "__main__":
    run()
