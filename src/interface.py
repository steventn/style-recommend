import os
import timm
import torch
from torchvision import transforms as T
from PIL import Image
from src.dataset_handler import DatasetHandler


def get_recommendations(image):
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    root = os.path.abspath(os.path.join(curr_dir, ".."))
    data_dir = os.path.join(root, "data")
    device = "cpu"
    mean, std, im_size = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], 224
    transformations = T.Compose([T.Resize((im_size, im_size)), T.ToTensor(), T.Normalize(mean=mean, std=std)])

    _, _, _, sub_cat_classes = DatasetHandler.create_data_loaders(root=os.path.join(data_dir), label_name="subCategory",
                                                   transformations=transformations,
                                                   batch_size=32)
    _, _, _, sub_color_classes = DatasetHandler.create_data_loaders(root=os.path.join(data_dir), label_name="baseColour",
                                                     transformations=transformations,
                                                     batch_size=32)

    cat_model = timm.create_model("rexnet_150", pretrained=False, num_classes=len(sub_cat_classes))
    color_model = timm.create_model("rexnet_150", pretrained=False, num_classes=len(sub_color_classes))

    cat_model_path = os.path.join(root, "models", "ecommerce_best_model_subcat1.pth")
    color_model_path = os.path.join(root, "models", "ecommerce_best_model_color.pth")

    cat_model.load_state_dict(torch.load(cat_model_path, map_location=device))
    color_model.load_state_dict(torch.load(color_model_path, map_location=device))

    cat_model = cat_model.to(device)
    color_model = color_model.to(device)

    cat_model.eval()
    color_model.eval()

    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])

    new_image = Image.open(image).convert("RGB")
    input_data = transform(new_image).unsqueeze(0).to(device)

    with torch.no_grad():
        cat_predictions = cat_model(input_data)

    with torch.no_grad():
        color_predictions = color_model(input_data)

    cat_probs = torch.nn.functional.softmax(cat_predictions, dim=1)
    color_probs = torch.nn.functional.softmax(color_predictions, dim=1)

    predicted_cat_class_index = torch.argmax(cat_probs, dim=1).item()
    predicted_color_class_index = torch.argmax(color_probs, dim=1).item()

    predicted_cat_class_name = list(sub_cat_classes.keys())[predicted_cat_class_index]
    predicted_color_class_name = list(sub_color_classes.keys())[predicted_color_class_index]

    print(f"Predicted Category Class: {predicted_cat_class_name}")
    print(f"Predicted Color Class: {predicted_color_class_name}")

    response = {
        "predicted_color": predicted_color_class_name,
        "predicted_category": predicted_cat_class_name
    }

    return response
