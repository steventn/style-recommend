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
    category_model_path = os.path.join(model_path, 'ecommerce_best_model_subcategory.pth')

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
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    root = os.path.abspath(os.path.join(curr_dir, ".."))
    data_dir = os.path.join(root, "data")
    device = "cpu"
    mean, std, im_size = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], 224
    transformations = T.Compose([T.Resize((im_size, im_size)), T.ToTensor(), T.Normalize(mean=mean, std=std)])

    # Load the class names
    _, _, _, sub_cat_classes = DatasetHandler.create_data_loaders(root=os.path.join(data_dir), label_name="subCategory",
                                                   transformations=transformations,
                                                   batch_size=32)
    _, _, _, sub_color_classes = DatasetHandler.create_data_loaders(root=os.path.join(data_dir), label_name="baseColour",
                                                     transformations=transformations,
                                                     batch_size=32)

    # Load the trained model
    cat_model = timm.create_model("rexnet_150", pretrained=False, num_classes=len(sub_cat_classes))
    color_model = timm.create_model("rexnet_150", pretrained=False, num_classes=len(sub_color_classes))

    # Specify the path to the trained model file
    cat_model_path = os.path.join(root, "models", "ecommerce_best_model_subcat1.pth")
    color_model_path = os.path.join(root, "models", "ecommerce_best_model_color.pth")

    # Load the trained weights
    cat_model.load_state_dict(torch.load(cat_model_path, map_location=device))
    color_model.load_state_dict(torch.load(color_model_path, map_location=device))

    # Move the model to the device (GPU or CPU)
    cat_model = cat_model.to(device)
    color_model = color_model.to(device)

    # Set the model to evaluation mode
    cat_model.eval()
    color_model.eval()

    # Define the data preprocessing transformation
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])

    # Load and preprocess the new image
    new_image = Image.open(image).convert("RGB")
    input_data = transform(new_image).unsqueeze(0).to(device)

    # Perform inference
    with torch.no_grad():
        # Forward pass
        cat_predictions = cat_model(input_data)

    # Perform inference
    with torch.no_grad():
        # Forward pass
        color_predictions = color_model(input_data)

    # Post-process the predictions as needed
    # Here, we assume that the model outputs class probabilities
    cat_probs = torch.nn.functional.softmax(cat_predictions, dim=1)
    color_probs = torch.nn.functional.softmax(color_predictions, dim=1)

    # Get the predicted class index
    predicted_cat_class_index = torch.argmax(cat_probs, dim=1).item()
    predicted_color_class_index = torch.argmax(color_probs, dim=1).item()

    # Map the class index to the original class name
    predicted_cat_class_name = list(sub_cat_classes.keys())[predicted_cat_class_index]
    predicted_color_class_name = list(sub_color_classes.keys())[predicted_color_class_index]

    print(f"Predicted Category Class: {predicted_cat_class_name}")
    print(f"Predicted Color Class: {predicted_color_class_name}")

    response = {
        "predicted_color": predicted_color_class_name,
        "predicted_category": predicted_cat_class_name
    }

    return response


if __name__ == "__main__":
    run()
