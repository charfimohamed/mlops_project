from fastapi import FastAPI
from http import HTTPStatus
import torch
from src.models.model import CatDogModel
from fastapi import UploadFile, File
import cv2
from fastapi.responses import FileResponse
from torchvision import transforms
from pathlib import Path
from PIL import Image
import os


app = FastAPI()

@app.post("/cv_model/")
async def cv_model(data: UploadFile = File(...)):
   with open('image.jpg', 'wb') as image:
      content = await data.read()
      image.write(content)
      image.close()

   model = CatDogModel()
   # loading the best parameters for the model
   try:
    checkpoint = torch.load(Path("../src/models/checkpoints/model_best_checkpoint.pth"))
   except FileNotFoundError:
    checkpoint = torch.load(Path("checkpoints/model_best_checkpoint.pth"))
   model.load_state_dict(checkpoint["model"])
   model.eval()
   image_size = model.im_size
    # add transformations to images and converting images into tensors
   img = cv2.imread("image.jpg")
   img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) #Convert from numpy ndarray to PIL Image

   data_resize = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
   img = data_resize(img)
   output = model(img[None,...])
   _, pred = torch.max(output, dim=1)
   class_labels = {0: "cat", 1: "dog"}
   
   response = {
      "input": data,
      "message": HTTPStatus.OK.phrase,
      "status-code": HTTPStatus.OK,
      "prediction for the uploaded image": class_labels[int(pred.item())]
      
   }
   return response
