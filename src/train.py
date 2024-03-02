import os, torch, shutil, numpy as np, pandas as pd
from glob import glob
from PIL import Image
from torch.utils.data import random_split, Dataset, DataLoader
from torchvision import transforms as T

torch.manual_seed(2024)


class CustomDataset(Dataset):

    def __init__(self, root, transformations=None):
        self.transformations, self.root = transformations, root
        all_img_paths = os.path.join(root, "e-commerce", "images", "*")
        self.img_paths = sorted(glob(all_img_paths))
        data_path = os.path.join(root, "styles.csv")
        data = pd.read_csv(data_path)
        ids = list(data["id"])
        label = list(data["subCategory"])

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
    ts_len = total_len - (train_length + val_length)

    train_dataset, val_dataset, test_dataset = random_split(dataset=dataset, lengths=[train_length, val_length, ts_len])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, dataset.cls_names


def main():
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    root = os.path.abspath(os.path.join(curr_dir, ".."))
    data_dir = os.path.join(root, "data")

    mean, std, im_size = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], 224
    transformations = T.Compose([T.Resize((im_size, im_size)), T.ToTensor(), T.Normalize(mean=mean, std=std)])

    train_data_loader, val_data_loader, test_data_loader, classes = create_data_loaders(root=os.path.join(data_dir),
                                                                                        transformations=transformations,
                                                                                        batch_size=32)

    print(f"Train DataLoader length: {len(train_data_loader)}")
    print(f"Validation DataLoader length: {len(val_data_loader)}")
    print(f"Test DataLoader length: {len(test_data_loader)}")
    print(f"Classes: {classes}")


if __name__ == "__main__":
    main()
