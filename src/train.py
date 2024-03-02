import os, torch, shutil, numpy as np, pandas as pd
from glob import glob
from PIL import Image
from torch.utils.data import random_split, Dataset, DataLoader
from torchvision import transforms as T

torch.manual_seed(2024)


class CustomDataset(Dataset):

    def __init__(self, root, transformations=None):
        self.transformations, self.root = transformations, root
        self.img_paths = sorted(glob(os.path.join(root, "e-commerce", "images", "*")))
        data_path = os.path.join(root, "styles.csv")
        print(f"Attempting to read data from: {data_path}")
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


def get_dls(root, transformations, bs, split=[0.9, 0.05, 0.05], ns=4):
    ds = CustomDataset(root=root, transformations=transformations)

    total_len = len(ds)
    tr_len = int(total_len * split[0])
    vl_len = int(total_len * split[1])
    ts_len = total_len - (tr_len + vl_len)

    tr_ds, vl_ds, ts_ds = random_split(dataset=ds, lengths=[tr_len, vl_len, ts_len])

    tr_dl, val_dl, ts_dl = DataLoader(tr_ds, batch_size=bs, shuffle=True, num_workers=ns), DataLoader(vl_ds,
                                                                                                      batch_size=bs,
                                                                                                      shuffle=False,
                                                                                                      num_workers=ns), DataLoader(
        ts_ds, batch_size=1, shuffle=False, num_workers=ns)

    return tr_dl, val_dl, ts_dl, ds.cls_names


def main():
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    root = os.path.abspath(os.path.join(curr_dir, ".."))
    data_dir = os.path.join(root, "data")

    mean, std, im_size = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], 224
    tfs = T.Compose([T.Resize((im_size, im_size)), T.ToTensor(), T.Normalize(mean=mean, std=std)])

    tr_dl, val_dl, ts_dl, classes = get_dls(root=os.path.join(data_dir), transformations=tfs, bs=32)

    print(f"Train DataLoader length: {len(tr_dl)}")
    print(f"Validation DataLoader length: {len(val_dl)}")
    print(f"Test DataLoader length: {len(ts_dl)}")
    print(f"Classes: {classes}")


if __name__ == "__main__":
    main()
