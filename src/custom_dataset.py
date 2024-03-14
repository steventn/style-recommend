import os
import pandas as pd
from torch.utils.data import Dataset
from glob import glob
from PIL import Image


class CustomDataset(Dataset):

    def __init__(self, root, label_name, transformations=None):
        self.transformations, self.root = transformations, root
        all_img_paths = os.path.join(root, "e-commerce", "images", "*")
        self.img_paths = sorted(glob(all_img_paths))
        data_path = os.path.join(root, "styles.csv")
        data = pd.read_csv(data_path)
        ids = list(data["id"])
        label = list(data[f"{label_name}"])

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
