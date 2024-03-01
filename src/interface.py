import torch
from torchvision import transforms
from PIL import Image

from src.train import CustomModel, num_classes, labels_df

# Load the trained model
model = CustomModel(num_classes)
model.load_state_dict(torch.load('models/your_model.pth'))
model.eval()


# Preprocess an image for inference
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path)
    input_tensor = transform(image).unsqueeze(0)
    return input_tensor


# Make a prediction
image_path = 'path/to/your/image.jpg'
input_tensor = preprocess_image(image_path)
with torch.no_grad():
    output = model(input_tensor)
    predicted_class = torch.argmax(output).item()

# Print the predicted class
print(f"Predicted class: {predicted_class}, Class name: {labels_df['article_type'].unique()[predicted_class]}")
