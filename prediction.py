import os
import ssl
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from matplotlib import transforms
from torchvision import transforms
import torchvision

# Turn of SSL verification
ssl._create_default_https_context = ssl._create_unverified_context

# Define transformations for test data
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)

# Load the trained model
model = torchvision.models.efficientnet_b0()
num_ftrs = 1280
model.classifier = nn.Sequential(nn.Linear(num_ftrs, 1), nn.Sigmoid())

# Load the weights from the trained model
model.load_state_dict(torch.load("agar3000.pth", map_location=torch.device("cpu")))

# Prepare test data
test_data_dir = "test_data"
test_image_paths = [
    os.path.join(test_data_dir, filename)
    for filename in os.listdir(test_data_dir)
    if filename.endswith(".jpg")
]

# Perform predictions on the test data
device = torch.device("cpu")
model = model.to(device)
model.eval()

# Store predictions
predictions = []
for image_path in test_image_paths:
    image_id = os.path.basename(image_path)  # Extract file ID
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # Add batch dimension
    image = image.to(device)
    output = model(image)
    threshold = 0.5
    predicted = (output >= threshold).int().item()
    predictions.append({"ID": image_id, "TARGET": predicted})


# Create a DataFrame from predictions and save it to file
df = pd.DataFrame(predictions)
df.to_csv("predictions.csv", index=False)
