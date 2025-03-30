import torch
import os
import ast
import numpy as np
from PIL import Image
from typing import List
from fastapi import FastAPI
from pydantic import BaseModel
from enum import Enum
from torchvision import transforms

class Payload(BaseModel):
    input: List[List[List[float]]]

# Returns the class name by index
def class_names(class_idx)->str:
    class_names = {
        0 : "Aphid",
        1 : "Armyworm",
        2 : "Cutworm",
        3 : "Diamondback_moth",
        4 : "Flea_beatle"
    }
    return class_names[class_idx]

# Load the model from the directory
def load_model(model_dir):
    model = torch.jit.load(os.path.join(model_dir, 'torch_script_vit.pt'))
    model.to("cuda") if torch.cuda.is_available() else model.to("cpu")
    return model

# Receives a tensor and a model
def predict(preprocessed_image, model):
    preprocessed_image = preprocessed_image.unsqueeze(0)
    if torch.cuda.is_available():
        preprocessed_image = preprocessed_image.to("cuda")
    pred_logits = model(preprocessed_image)
    pred_probs = torch.softmax(pred_logits, dim=1)
    probs, class_idx = torch.max(pred_probs, dim=1)
    output = {'class' : class_names(class_idx.detach().cpu().numpy().item()),
              'class_idx' : class_idx.detach().cpu().numpy().item(),
              'probability': probs.detach().cpu().numpy().item()}
    return output

# def preprocess(image):
#     image = np.array(image)  # Ensure it's a NumPy array
#     if image.shape != (3, 224, 224):  # Check if shape is correct
#         raise ValueError(f"Expected input shape (3, 224, 224), but got {image.shape}")
    
#     image = torch.tensor(image, dtype=torch.float32)  # Convert to Tensor
#     transform = transforms.Compose([
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])
#     image = transform(image)
#     return image

def preprocess(image):
    image = np.array(image)
    if image.shape != (3, 224, 224):
        raise ValueError(f"Expected input shape (3, 224, 224), but got {image.shape}")
    image = torch.tensor(image, dtype=torch.float32)
    transform = transforms.Compose([
        transforms.Resize(size=(224, 224)),  # Resize should come before ToTensor()
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image = transform(image)
    return image

# def load_payload(payload: Payload):
#     return np.array(payload.input, dtype=np.float32)  # Convert directly to NumPy array

def load_payload(payload: Payload):
    payload = payload.json()
    payload = ast.literal_eval(payload)
    image = payload['input']
    return image

app = FastAPI()
model = load_model(".")

@app.get('/ping')
def pint():
    return "pong"

@app.post('/invocations')
def invoke(payload: Payload):
    image = load_payload(payload)
    image = preprocess(image)
    output = predict(image, model)
    return output