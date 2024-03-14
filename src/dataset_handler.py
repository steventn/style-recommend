import numpy as np
import random
from torch.utils.data import random_split, DataLoader
from matplotlib import pyplot as plt
from torchvision import transforms as T
from src.custom_dataset import CustomDataset


class DatasetHandler:
    @staticmethod
    def create_data_loaders(root, label_name, transformations, batch_size, split_ratio=[0.9, 0.05, 0.05], num_workers=4):
        dataset = CustomDataset(root=root, label_name=label_name, transformations=transformations)

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

    @staticmethod
    def tensor_to_image(tensor, image_type="rgb"):
        gray_transforms = T.Compose([T.Normalize(mean=[0.], std=[1 / 0.5]), T.Normalize(mean=[-0.5], std=[1])])
        rgb_transforms = T.Compose([T.Normalize(mean=[0., 0., 0.], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
                                    T.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.])])

        inverse_transform = gray_transforms if image_type == "gray" else rgb_transforms

        return (inverse_transform(tensor) * 255).detach().squeeze().cpu().permute(1, 2, 0).numpy().astype(
            np.uint8) if image_type == "gray" else (inverse_transform(tensor) * 255).detach().cpu().permute(1, 2,
                                                                                                            0).numpy().astype(
            np.uint8)

    @staticmethod
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
                plt.imshow(DatasetHandler.tensor_to_image(image, color_map), cmap=color_map)
            else:
                plt.imshow(DatasetHandler.tensor_to_image(image))

            plt.axis('off')
            if class_names is not None:
                plt.title(f"Ground Truth -> {class_names[int(ground_truth)]}")
            else:
                plt.title(f"Ground Truth -> {ground_truth}")
        plt.show()

