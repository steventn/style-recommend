import os, torch, numpy as np, pandas as pd
import random
import timm
import torchmetrics

from glob import glob
from PIL import Image
from torch.utils.data import random_split, Dataset, DataLoader
from torchvision import transforms as T
from matplotlib import pyplot as plt
from tqdm import tqdm

torch.manual_seed(2024)


class CustomDataset(Dataset):

    def __init__(self, root, transformations=None):
        self.transformations, self.root = transformations, root
        all_img_paths = os.path.join(root, "e-commerce", "images", "*")
        self.img_paths = sorted(glob(all_img_paths))
        data_path = os.path.join(root, "styles.csv")
        data = pd.read_csv(data_path)
        ids = list(data["id"])
        label = list(data["baseColour"])

        self.ids, self.label = [], []
        self.cls_names, self.cls_counts, count, data_count = {}, {}, 0, 0
        for idx, (id, class_name) in enumerate(zip(ids, label)):
            self.ids.append(id)
            self.label.append(class_name)
            if class_name not in self.cls_names:
                self.cls_names[class_name] = count
                self.cls_counts[class_name] = 1
                count += 1
            else:
                self.cls_counts[class_name] += 1

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.root, "e-commerce", "images", f"{self.ids[idx]}.jpg")).convert("RGB")
        true_label = self.cls_names[self.label[idx]]

        if self.transformations is not None: img = self.transformations(img)

        return img, true_label


def create_data_loaders(root, transformations, batch_size, split_ratio=[0.9, 0.05, 0.05], num_workers=4):
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

    return train_loader, val_loader, test_loader, dataset.cls_names


def tensor_to_image(tensor, image_type="rgb"):
    gray_transforms = T.Compose([T.Normalize(mean=[0.], std=[1 / 0.5]), T.Normalize(mean=[-0.5], std=[1])])
    rgb_transforms = T.Compose([T.Normalize(mean=[0., 0., 0.], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
                                T.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.])])

    inverse_transform = gray_transforms if image_type == "gray" else rgb_transforms

    return (inverse_transform(tensor) * 255).detach().squeeze().cpu().permute(1, 2, 0).numpy().astype(
        np.uint8) if image_type == "gray" else (inverse_transform(tensor) * 255).detach().cpu().permute(1, 2,
                                                                                                        0).numpy().astype(
        np.uint8)


def visualize_dataset(data, num_images, rows, color_map=None, class_names=None):
    assert color_map in ["rgb", "gray"], "Specify whether the image is grayscale or color!"
    if color_map == "rgb":
        color_map = "viridis"

    plt.figure(figsize=(20, 10))
    indices = [random.randint(0, len(data) - 1) for _ in range(num_images)]

    for idx, index in enumerate(indices):
        image, ground_truth = data[index]
        # Start plot
        plt.subplot(rows, num_images // rows, idx + 1)

        if color_map:
            plt.imshow(tensor_to_image(image, color_map), cmap=color_map)
        else:
            plt.imshow(tensor_to_image(image))

        plt.axis('off')
        if class_names is not None:
            plt.title(f"Ground Truth -> {class_names[int(ground_truth)]}")
        else:
            plt.title(f"Ground Truth -> {ground_truth}")
    plt.show()


def setup_training(model, device):
    # Set up training configuration
    # Running on CPU due to hardware incompatibility
    return model.to(device).eval(), 5, device, torch.nn.CrossEntropyLoss(), torch.optim.Adam(params=model.parameters(),
                                                                                             lr=3e-4)


def move_to_device(batch, device):
    # Move data to the specified device
    return batch[0].to(device), batch[1].to(device)


def calculate_metrics(model, images, targets, loss_function, epoch_loss, epoch_accuracy, epoch_f1, f1_score):
    # Calculate metrics during training
    predictions = model(images)
    loss = loss_function(predictions, targets)
    return loss, epoch_loss + loss.item(), epoch_accuracy + (
            torch.argmax(predictions, dim=1) == targets).sum().item(), epoch_f1 + f1_score(predictions, targets)


def train_model(classes, train_data_loader, val_data_loader, device, root):
    save_prefix, save_dir = "ecommerce", os.path.join(root, "models")
    model = timm.create_model("rexnet_150", pretrained=True, num_classes=len(classes)).to(device)
    model, epochs, device, loss_function, optimizer = setup_training(model, device)
    f1_score_metric = torchmetrics.F1Score(task="multiclass", num_classes=len(classes)).to(device)
    print("Start training...")
    best_accuracy, best_loss, threshold, not_improved, patience = 0, float("inf"), 0.01, 0, 5
    train_losses, val_losses, train_accuracies, val_accuracies, train_f1_scores, val_f1_scores = [], [], [], [], [], []

    best_loss = float(torch.inf)

    # Training loop
    for epoch in range(epochs):
        # Training phase
        epoch_loss, epoch_accuracy, epoch_f1 = 0, 0, 0
        for idx, batch in tqdm(enumerate(train_data_loader)):
            images, targets = move_to_device(batch, device)

            loss, epoch_loss, epoch_accuracy, epoch_f1 = calculate_metrics(model, images, targets, loss_function,
                                                                           epoch_loss, epoch_accuracy, epoch_f1,
                                                                           f1_score_metric)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Calculate and store training metrics
        train_loss_to_track = epoch_loss / len(train_data_loader)
        train_accuracy_to_track = epoch_accuracy / len(train_data_loader.dataset)
        train_f1_to_track = epoch_f1 / len(train_data_loader)
        train_losses.append(train_loss_to_track)
        train_accuracies.append(train_accuracy_to_track)
        train_f1_scores.append(train_f1_to_track)

        print(f"{epoch + 1}-epoch train process is completed!")
        print(f"{epoch + 1}-epoch train loss          -> {train_loss_to_track:.3f}")
        print(f"{epoch + 1}-epoch train accuracy      -> {train_accuracy_to_track:.3f}")
        print(f"{epoch + 1}-epoch train f1-score      -> {train_f1_to_track:.3f}")

        # Validation phase
        model.eval()
        with torch.no_grad():
            val_epoch_loss, val_epoch_accuracy, val_epoch_f1 = 0, 0, 0
            for idx, batch in enumerate(val_data_loader):
                images, targets = move_to_device(batch, device)
                loss, val_epoch_loss, val_epoch_accuracy, val_epoch_f1 = calculate_metrics(model, images, targets,
                                                                                           loss_function,
                                                                                           val_epoch_loss,
                                                                                           val_epoch_accuracy,
                                                                                           val_epoch_f1,
                                                                                           f1_score_metric)

            # Calculate and store validation metrics
            val_loss_to_track = val_epoch_loss / len(val_data_loader)
            val_accuracy_to_track = val_epoch_accuracy / len(val_data_loader.dataset)
            val_f1_to_track = val_epoch_f1 / len(val_data_loader)
            val_losses.append(val_loss_to_track)
            val_accuracies.append(val_accuracy_to_track)
            val_f1_scores.append(val_f1_to_track)

            print(f"{epoch + 1}-epoch validation process is completed!")
            print(f"{epoch + 1}-epoch validation loss     -> {val_loss_to_track:.3f}")
            print(f"{epoch + 1}-epoch validation accuracy -> {val_accuracy_to_track:.3f}")
            print(f"{epoch + 1}-epoch validation f1-score -> {val_f1_to_track:.3f}")

            # Check if validation loss improved
            if val_loss_to_track < (best_loss + threshold):
                os.makedirs(save_dir, exist_ok=True)
                best_loss = val_loss_to_track
                torch.save(model.state_dict(), f"{save_dir}/{save_prefix}_best_model_color.pth")
            else:
                not_improved += 1
                print(f"Loss value did not decrease for {not_improved} epochs")
                if not_improved == patience:
                    print(f"Stop training since loss value did not decrease for {patience} epochs.")
                    break


def main():
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    root = os.path.abspath(os.path.join(curr_dir, ".."))
    data_dir = os.path.join(root, "data")
    device = "cpu"

    mean, std, im_size = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], 224
    transformations = T.Compose([T.Resize((im_size, im_size)), T.ToTensor(), T.Normalize(mean=mean, std=std)])

    train_data_loader, val_data_loader, test_data_loader, classes = create_data_loaders(root=os.path.join(data_dir),
                                                                                        transformations=transformations,
                                                                                        batch_size=32)

    print(f"Train DataLoader length: {len(train_data_loader)}")
    print(f"Validation DataLoader length: {len(val_data_loader)}")
    print(f"Test DataLoader length: {len(test_data_loader)}")
    print(f"Classes: {classes}")

    visualize_dataset(train_data_loader.dataset, 20, 4, "rgb", list(classes.keys()))
    visualize_dataset(val_data_loader.dataset, 20, 4, "rgb", list(classes.keys()))
    visualize_dataset(test_data_loader.dataset, 20, 4, "rgb", list(classes.keys()))

    train_model(classes, train_data_loader, val_data_loader, device, root)


if __name__ == "__main__":
    main()
