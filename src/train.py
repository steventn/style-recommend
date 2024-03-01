import os
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim

from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from PIL import Image


class CustomDataset(Dataset):
    def __init__(self, labels_df, root_dir, transform=None):
        self.labels_df = labels_df
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        img_name = str(self.labels_df.iloc[idx, 0])
        img_path = os.path.join(script_dir, '..', 'data', 'e-commerce', 'images', str(img_name) + '.jpg')
        image = Image.open(img_path)

        # Extract the label columns
        gender = self.labels_df.iloc[idx, 1]
        master_category = self.labels_df.iloc[idx, 2]
        sub_category = self.labels_df.iloc[idx, 3]
        article_type = self.labels_df.iloc[idx, 4]
        base_color = self.labels_df.iloc[idx, 5]
        season = self.labels_df.iloc[idx, 6]
        year = self.labels_df.iloc[idx, 7]
        usage = self.labels_df.iloc[idx, 8]
        display_name = self.labels_df.iloc[idx, 9]

        label = {
            'gender': gender,
            'master_category': master_category,
            'sub_category': sub_category,
            'article_type': article_type,
            'base_color': base_color,
            'season': season,
            'year': year,
            'usage': usage,
            'display_name': display_name
        }

        if self.transform:
            image = self.transform(image)

        return image, label


def collate_fn(batch):
    images, labels = zip(*batch)

    # No need to apply ToTensor() here as images are already transformed in CustomDataset
    # Resize images to a common size if needed
    images = [data_transform(image) for image in images]

    # Stack images and handle varying channel sizes
    max_channels = max(image.shape[0] for image in images)
    stacked_images = torch.stack(
        [torch.cat([image, torch.zeros(max_channels - image.shape[0], *image.shape[1:])]) for image in images])

    return stacked_images, labels


# Load the CSV file containing labels
script_dir = os.path.dirname(os.path.realpath(__file__))
csv_file_path = os.path.join(script_dir, '..', 'data', 'styles.csv')
labels_df = pd.read_csv(csv_file_path)

# Create an instance of your custom dataset
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

data_folder = os.path.join('data', 'e-commerce', 'images')
custom_dataset = CustomDataset(labels_df, os.path.join(script_dir, data_folder), transform=data_transform)

# Create data loader
train_loader = DataLoader(custom_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)


class CustomModel(nn.Module):
    def __init__(self, num_classes):
        super(CustomModel, self).__init__()
        self.base_model = models.resnet18(pretrained=True)
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, num_classes)

    def forward(self, x):
        return self.base_model(x)


# Create an instance of your model
num_classes = len(labels_df['articleType'].unique())
model = CustomModel(num_classes)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Instantiate label encoder
label_encoder = LabelEncoder()

# Label encode the target variable outside the training loop
labels_df['encoded_article_type'] = label_encoder.fit_transform(labels_df['articleType'])

# Create an instance of your model
num_classes = len(label_encoder.classes_)
model = CustomModel(num_classes)

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    total_loss = 0.0
    for inputs, labels in train_loader:
        inputs, target = inputs.to(device), labels['encoded_article_type'].to(device)

        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {average_loss:.4f}')

# Save the trained model
torch.save(model.state_dict(), 'models/your_model.pth')
