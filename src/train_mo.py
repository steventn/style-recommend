import os, torch, numpy as np, pandas as pd
import random

import pretrainedmodels
import torchmetrics
import torch.nn.functional as F

from glob import glob
from PIL import Image
from torch.utils.data import random_split, Dataset, DataLoader
from torch import nn
from torchvision import transforms as T
from matplotlib import pyplot as plt
from tqdm import tqdm

torch.manual_seed(2024)
curr_dir = os.path.dirname(os.path.realpath(__file__))
root = os.path.abspath(os.path.join(curr_dir, ".."))


class CustomDataset(Dataset):

    def __init__(self, root, transformations=None):
        self.transformations, self.root = transformations, root
        all_img_paths = os.path.join(root, "e-commerce", "images", "*")
        self.img_paths = sorted(glob(all_img_paths))
        data_path = os.path.join(root, "styles.csv")
        data = pd.read_csv(data_path)
        ids = list(data["id"])
        sub_category_labels = list(data["subCategory"])
        base_color_labels = list(data["baseColour"])

        self.ids, self.sub_category_labels, self.base_color_labels = [], [], []
        self.sub_category_names, self.sub_category_counts, sub_category_count = {}, {}, 0
        self.base_color_names, self.base_color_counts, base_color_count = {}, {}, 0

        for idx, (id, sub_category, base_color) in enumerate(zip(ids, sub_category_labels, base_color_labels)):
            self.ids.append(id)
            self.sub_category_labels.append(sub_category)
            self.base_color_labels.append(base_color)

            # Handling subCategory (class_name)
            if sub_category not in self.sub_category_names:
                self.sub_category_names[sub_category] = sub_category_count
                self.sub_category_counts[sub_category] = 1
                sub_category_count += 1
            else:
                self.sub_category_counts[sub_category] += 1

            # Handling baseColor
            if base_color not in self.base_color_names:
                self.base_color_names[base_color] = base_color_count
                self.base_color_counts[base_color] = 1
                base_color_count += 1
            else:
                self.base_color_counts[base_color] += 1

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.root, "e-commerce", "images", f"{self.ids[idx]}.jpg")).convert("RGB")
        sub_category_label = self.sub_category_names[self.sub_category_labels[idx]]
        base_color_label = self.base_color_names[self.base_color_labels[idx]]

        if self.transformations is not None:
            img = self.transformations(img)

        return img, sub_category_label, base_color_label


class CNN1(nn.Module):
    def __init__(self, num_sub_categories, num_base_colors):
        super(CNN1, self).__init__()
        self.base_model = pretrainedmodels.__dict__["resnet34"](pretrained="imagenet")

        with torch.no_grad():
            x = torch.randn(1, 3, 224, 224)
            out = self.base_model.features(x)
            self.num_features = out.shape[1]

        self.classification_head_sub_category = nn.Linear(self.num_features, num_sub_categories)
        self.classification_head_base_color = nn.Linear(self.num_features, num_base_colors)

    def forward(self, x):
        if isinstance(x, tuple):
            # If x is a tuple of images, process each image separately
            features = [self.base_model.features(img) for img in x]
            features = [F.adaptive_avg_pool2d(feature, 1).reshape(feature.size(0), -1) for feature in features]
            features = torch.cat(features, dim=1)
        else:
            # If x is a single image, process it normally
            features = self.base_model.features(x)
            features = F.adaptive_avg_pool2d(features, 1).reshape(features.size(0), -1)

        logits_sub_category = self.classification_head_sub_category(features)
        logits_base_color = self.classification_head_base_color(features)

        return logits_sub_category, logits_base_color


def create_data_loaders(root, transformations, batch_size, split_ratio=None, num_workers=4):
    if split_ratio is None:
        split_ratio = [0.9, 0.05, 0.05]
    dataset = CustomDataset(root=root, transformations=transformations)

    # Calculate dataset lengths for train, validation, and test sets
    total_len = len(dataset)
    train_length = int(total_len * split_ratio[0])
    val_length = int(total_len * split_ratio[1])
    test_length = total_len - (train_length + val_length)

    train_dataset, val_dataset, test_dataset = random_split(dataset=dataset,
                                                            lengths=[train_length, val_length, test_length])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, dataset.sub_category_names, dataset.base_color_names


def visualize_dataset(data, num_images, rows, color_map=None, class_names=None, color_names=None):
    assert color_map in ["rgb", "gray"], "Specify whether the image is grayscale or color!"
    if color_map == "rgb":
        color_map = "viridis"

    plt.figure(figsize=(20, 10))
    indices = [random.randint(0, len(data) - 1) for _ in range(num_images)]

    for idx, index in enumerate(indices):
        image, sub_category, base_color = data[index]

        # Start plot
        plt.subplot(rows, num_images // rows, idx + 1)

        if color_map:
            plt.imshow(tensor_to_image(image, color_map), cmap=color_map)
        else:
            plt.imshow(tensor_to_image(image))

        plt.axis('off')
        if class_names is not None:
            plt.title(f"Sub Category: {class_names[int(sub_category)]}\nBase Color: {color_names[int(base_color)]}")
        else:
            plt.title(f"Sub Category: {sub_category}\nBase Color: {base_color}")
    plt.show()


def tensor_to_image(tensor, image_type="rgb"):
    gray_transforms = T.Compose([T.Normalize(mean=[0.], std=[1 / 0.5]), T.Normalize(mean=[-0.5], std=[1])])
    rgb_transforms = T.Compose([T.Normalize(mean=[0., 0., 0.], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
                                T.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.])])

    inverse_transform = gray_transforms if image_type == "gray" else rgb_transforms

    return (inverse_transform(tensor) * 255).detach().squeeze().cpu().permute(1, 2, 0).numpy().astype(
        np.uint8) if image_type == "gray" else (inverse_transform(tensor) * 255).detach().cpu().permute(1, 2,
                                                                                                        0).numpy().astype(
        np.uint8)


def setup_training(device, num_sub_category_classes, num_base_color_classes, epochs):
    # Set up training configuration
    base_model = CNN1(num_sub_category_classes, num_base_color_classes)

    model = torch.nn.Sequential(
        base_model,
        base_model.classification_head_sub_category,
        base_model.classification_head_base_color
    ).to(device)

    # Assuming you want separate loss functions for sub-category and base color
    sub_category_loss_function = torch.nn.CrossEntropyLoss()
    base_color_loss_function = torch.nn.CrossEntropyLoss()

    return model, epochs, device, (sub_category_loss_function, base_color_loss_function), torch.optim.Adam(
        params=model.parameters(), lr=3e-4)


def move_to_device(batch, device):
    # Move data to the specified device
    return batch[0].to(device), batch[1].to(device), batch[2].to(device)


def calculate_metrics(model, images, targets_sub_category, targets_base_color, loss_function, epoch_loss,
                      epoch_accuracy_sub_category, epoch_accuracy_base_color, epoch_f1, f1_score):
    # Calculate metrics during training
    print(images.shape)
    print(images.dtype)

    features = model(images)
    logits_sub_category = model[1][0](features)  # Access the first linear layer in the ModuleList
    logits_base_color = model[1][1](features)  # Access the second linear layer

    # Calculate classification losses
    classification_loss_sub_category = loss_function(logits_sub_category, targets_sub_category)
    classification_loss_base_color = loss_function(logits_base_color, targets_base_color)

    # Update total loss
    loss = classification_loss_sub_category + classification_loss_base_color

    # Calculate f1 score for sub-category classification task
    classification_f1_sub_category = f1_score(logits_sub_category, targets_sub_category)

    return loss, epoch_loss + loss.item(), epoch_accuracy_sub_category + (
            torch.argmax(logits_sub_category,
                         dim=1) == targets_sub_category).sum().item(), epoch_accuracy_base_color + (
                         torch.argmax(logits_base_color, dim=1) == targets_base_color).sum().item(), \
                 epoch_f1 + classification_f1_sub_category


def train_model(sub_category, base_colors, train_data_loader, val_data_loader, device, root):
    save_prefix, save_dir = "ecommerce", os.path.join(root, "models")

    # Get model and loss functions from setup_training
    model, epochs, device, loss_function, optimizer = setup_training(device, len(sub_category), len(base_colors), epochs=1)

    f1_score_metric = torchmetrics.F1Score(task="multiclass", num_classes=len(sub_category)).to(device)

    print("Start training...")

    best_accuracy_sub_category, best_accuracy_base_color, best_loss, threshold, not_improved, patience = 0, 0, float(
        "inf"), 0.01, 0, 5
    train_losses, val_losses, train_accuracies_sub_category, train_accuracies_base_color, val_accuracies_sub_category, val_accuracies_base_color, train_f1_scores, val_f1_scores = [], [], [], [], [], [], [], []

    # Training loop
    for epoch in range(epochs):
        # Training phase
        epoch_loss, epoch_accuracy_sub_category, epoch_accuracy_base_color, epoch_f1 = 0, 0, 0, 0
        for idx, batch in tqdm(enumerate(train_data_loader)):
            images, targets_sub_category, targets_base_color = move_to_device(batch, device)

            loss, epoch_loss, epoch_accuracy_sub_category, epoch_accuracy_base_color, epoch_f1 = calculate_metrics(
                model, images, targets_sub_category, targets_base_color,
                loss_function, epoch_loss, epoch_accuracy_sub_category,
                epoch_accuracy_base_color, epoch_f1,
                f1_score_metric)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Calculate and store training metrics
        train_loss_to_track = epoch_loss / len(train_data_loader)
        train_accuracy_sub_category_to_track = epoch_accuracy_sub_category / len(train_data_loader.dataset)
        train_accuracy_base_color_to_track = epoch_accuracy_base_color / len(train_data_loader.dataset)
        train_f1_to_track = epoch_f1 / len(train_data_loader)
        train_losses.append(train_loss_to_track)
        train_accuracies_sub_category.append(train_accuracy_sub_category_to_track)
        train_accuracies_base_color.append(train_accuracy_base_color_to_track)
        train_f1_scores.append(train_f1_to_track)

        print(f"{epoch + 1}-epoch train process is completed!")
        print(f"{epoch + 1}-epoch train loss          -> {train_loss_to_track:.3f}")
        print(f"{epoch + 1}-epoch train sub-category accuracy -> {train_accuracy_sub_category_to_track:.3f}")
        print(f"{epoch + 1}-epoch train base color accuracy  -> {train_accuracy_base_color_to_track:.3f}")
        print(f"{epoch + 1}-epoch train f1-score      -> {train_f1_to_track:.3f}")

        # Validation phase
        model.eval()
        with torch.no_grad():
            val_epoch_loss, val_epoch_accuracy_sub_category, val_epoch_accuracy_base_color, val_epoch_f1 = 0, 0, 0, 0
            for idx, batch in enumerate(val_data_loader):
                images, targets_sub_category, targets_base_color = move_to_device(batch, device)
                loss, val_epoch_loss, val_epoch_accuracy_sub_category, val_epoch_accuracy_base_color, val_epoch_f1 = calculate_metrics(
                    model, images, targets_sub_category,
                    targets_base_color, loss_function,
                    val_epoch_loss,
                    val_epoch_accuracy_sub_category,
                    val_epoch_accuracy_base_color,
                    val_epoch_f1,
                    f1_score_metric)

            # Calculate and store validation metrics
            val_loss_to_track = val_epoch_loss / len(val_data_loader)
            val_accuracy_sub_category_to_track = val_epoch_accuracy_sub_category / len(val_data_loader.dataset)
            val_accuracy_base_color_to_track = val_epoch_accuracy_base_color / len(val_data_loader.dataset)
            val_f1_to_track = val_epoch_f1 / len(val_data_loader)
            val_losses.append(val_loss_to_track)
            val_accuracies_sub_category.append(val_accuracy_sub_category_to_track)
            val_accuracies_base_color.append(val_accuracy_base_color_to_track)
            val_f1_scores.append(val_f1_to_track)

            print(f"{epoch + 1}-epoch validation process is completed!")
            print(f"{epoch + 1}-epoch validation loss     -> {val_loss_to_track:.3f}")
            print(f"{epoch + 1}-epoch validation sub-category accuracy -> {val_accuracy_sub_category_to_track:.3f}")
            print(f"{epoch + 1}-epoch validation base color accuracy  -> {val_accuracy_base_color_to_track:.3f}")
            print(f"{epoch + 1}-epoch validation f1-score -> {val_f1_to_track:.3f}")

            # Check if validation loss improved
            if val_loss_to_track < (best_loss + threshold):
                os.makedirs(save_dir, exist_ok=True)
                best_loss = val_loss_to_track
                torch.save(model.state_dict(), f"{save_dir}/{save_prefix}_best_model.pth")
            else:
                not_improved += 1
                print(f"Loss value did not decrease for {not_improved} epochs")
                if not_improved == patience:
                    print(f"Stop training since loss value did not decrease for {patience} epochs.")
                    break


def main():
    data_dir = os.path.join(root, "data")
    device = "cpu"

    mean, std, im_size = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], 224
    transformations = T.Compose([T.Resize((im_size, im_size)), T.ToTensor(), T.Normalize(mean=mean, std=std)])

    # Load the dataset
    train_data_loader, val_data_loader, test_data_loader, sub_category, base_color = create_data_loaders(
        root=os.path.join(data_dir),
        transformations=transformations,
        batch_size=32)

    print(f"Train DataLoader length: {len(train_data_loader)}")
    print(f"Validation DataLoader length: {len(val_data_loader)}")
    print(f"Test DataLoader length: {len(test_data_loader)}")

    print(f"Classes (subCategory): {sub_category}")
    print(f"Classes (baseColor): {base_color}")

    visualize_dataset(train_data_loader.dataset, 20, 4, "rgb", list(sub_category.keys()), list(base_color.keys()))
    visualize_dataset(val_data_loader.dataset, 20, 4, "rgb", list(sub_category.keys()), list(base_color.keys()))
    visualize_dataset(test_data_loader.dataset, 20, 4, "rgb", list(sub_category.keys()), list(base_color.keys()))

    # Assuming you want to train the model for subCategory and baseColor
    train_model(sub_category, base_color, train_data_loader, val_data_loader, device, root)


# def predict_image():
#     data_dir = os.path.join(root, "data")
#     device = "cpu"
#     mean, std, im_size = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], 224
#     transformations = T.Compose([T.Resize((im_size, im_size)), T.ToTensor(), T.Normalize(mean=mean, std=std)])
#
#     # Load the class names for subCategory
#     _, _, _, sub_category_classes = create_data_loaders(root=os.path.join(data_dir),
#                                                         transformations=transformations,
#                                                         batch_size=32)
#
#     # Load the class names for baseColor
#     _, _, _, base_color_classes = create_data_loaders(root=os.path.join(data_dir),
#                                                       transformations=transformations,
#                                                       batch_size=32)
#
#     # Load the trained model
#     model = timm.create_model("rexnet_150", pretrained=False,
#                               num_classes=len(sub_category_classes) + len(base_color_classes) + 1)
#
#     # Specify the path to the trained model file
#     model_path = os.path.join(root, "models", "ecommerce_best_model.pth")
#
#     # Load the trained weights
#     model.load_state_dict(torch.load(model_path, map_location=device))
#
#     # Move the model to the device (GPU or CPU)
#     model = model.to(device)
#
#     # Set the model to evaluation mode
#     model.eval()
#
#     # Define the data preprocessing transformation
#     transform = T.Compose([
#         T.Resize((224, 224)),
#         T.ToTensor(),
#         T.Normalize(mean=mean, std=std),
#     ])
#
#     # Specify the path to the new image for inference
#     new_image_path = os.path.join(root, "input", "featured_product-kids_tee.jpg")
#
#     # Load and preprocess the new image
#     new_image = Image.open(new_image_path).convert("RGB")
#     input_data = transform(new_image).unsqueeze(0).to(device)
#
#     # Perform inference
#     with torch.no_grad():
#         # Forward pass
#         predictions = model(input_data)
#
#     # Post-process the predictions as needed
#     # Here, we assume that the model outputs class probabilities
#     probs = torch.nn.functional.softmax(predictions, dim=1)
#
#     # Get the predicted class index for subCategory
#     predicted_class_index = torch.argmax(probs, dim=1).item()
#     # Map the class index to the original class name for subCategory
#     predicted_class_name = list(sub_category_classes.keys())[predicted_class_index]
#
#     # Get the predicted baseColor (you may need to adjust this based on your model architecture)
#     predicted_base_color = "Unknown"
#
#     # Visualize the input image
#     plt.imshow(np.array(new_image))
#     plt.title(f"Predicted subCategory: {predicted_class_name}, Predicted baseColor: {predicted_base_color}")
#     plt.axis('off')
#     plt.show()


if __name__ == "__main__":
    main()
