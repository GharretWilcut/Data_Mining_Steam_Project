# Convolutional Neural Network:
# Reads images from Steam API and uses ResNet50, a pretrained
# model, to extract embeddings which are saved to embeddings.csv.

import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pandas as pd
from tqdm import tqdm

IMAGE_FOLDER = "data/images" # Pulls from a folder of images in data folder
OUTPUT_FILE = "data/embeddings.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


##############
# TRANSFORMS #
##############
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean = [0.485, 0.456, 0.406],
        std = [0.229, 0.224, 0.225]
    )
])


#################
# LOADING MODEL #
#################
# Model Resnet50
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

# Remove final classification layer
model = nn.Sequential(*list(model.children())[:-1])

model.eval()
model.to(DEVICE)

##################
# PROCESS IMAGES #
##################
data = []

for filename in tqdm(os.listdir(IMAGE_FOLDER)):
    if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    path = os.path.join(IMAGE_FOLDER, filename)

    try:
        image = Image.open(path).convert("RGB")
        image = transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            embedding = model(image)
        
        embedding = embedding.squeeze().cpu().numpy()

        # Sets the filemame as the ID
        game_id = os.path.splitext(filename)[0]

        row = [game_id] + embedding.tolist()
        data.append(row)
    
    except Exception as e:
        print(f"Error processing {filename}: {e}")
    

###############
# SAVE TO CSV #
###############
columns = ["game_id"] + [f"f{i}" for i in range(len(data[0]) - 1)]
df = pd.DataFrame(data, columns=columns)
df.to_csv(OUTPUT_FILE, index=False)

print(f"Saved embeddings to {OUTPUT_FILE}")
