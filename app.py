import torch
import os
import io
import ast
import numpy as np
from PIL import Image
from typing import List
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from enum import Enum
from torchvision import transforms

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

# Receives and actual image an convert it into tensor
async def preprocess(file: UploadFile = File(...)):  
    contents = await file.read()

    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except:
        raise ValueError("The uploaded file could not be identified as an image.")
    
    transform = transforms.Compose([
        transforms.Resize(size=(224, 224)),  # Resize should come before ToTensor()
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image)
    return image_tensor

app = FastAPI()
model = load_model(".")

@app.get('/ping')
def pint():
    return "pong"

@app.post('/invocations')
async def invoke(file: UploadFile = File(...)):
    image = await preprocess(file)
    output = predict(image, model)
    return output