from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, Request
from typing import List
from models.googlenet_augmented import train_and_get_accuracy as gn_model
from models.read_accuracy import read_accuracy
from models.preprocess import preprocess
from fastapi.middleware.cors import CORSMiddleware
from models.alexnet import train_and_get_accuracy as alexnet_model
from models.mobilenet import train_and_get_accuracy_mobilenet as mobile_net
from models.vgg16 import train_and_get_accuracy_vgg16 as vgg16
app = FastAPI()

origins = [
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ModelInput(BaseModel):
    email: str
    no_of_classes: int

UPLOADS_DIR = "uploads"

@app.get("/")
def hey():
    return {"hey": "bye"}

@app.get("/run_model/{email}")
async def run_model(email: str):
    a = await read_accuracy(email)
    return {"accuracy": a}

@app.post("/run_model/{email}")
async def handle_model(email: str, input_data: ModelInput, request: Request):
    email = input_data.email
    no_of_classes = input_data.no_of_classes
    selected_models = await request.json()  # Access JSON data from request body
    # a = await preprocess(email, no_of_classes)
    accuracy=[]
    for model in selected_models["selectedModels"]:
        if model == "GoogleNet":
            result = await gn_model(email, no_of_classes)
            print("GoogleNet Model")
            print(result["train_accuracy"])
            print(result["test_accuracy"])
            print(result["dev_accuracy"])
            accuracy.append({"GoogleNet": result})
        elif model == "AlexNet":
            result = await alexnet_model(email, no_of_classes)
            print("AlexNet Model")
            print(result["train_accuracy"])
            print(result["test_accuracy"])
            print(result["dev_accuracy"])
            accuracy.append({"AlexNet": result})
        elif model =="VGG16":
            result = await vgg16(email, no_of_classes)
            print("VGG16 Model")
            print(result["train_accuracy"])
            print(result["test_accuracy"])
            print(result["dev_accuracy"])
            accuracy.append({"VGG16": result})
        elif model =="MobileNet":
            result = await mobile_net(email, no_of_classes)
            print("MobileNetV2 Model")
            print(result["train_accuracy"])
            print(result["test_accuracy"])
            print(result["dev_accuracy"])
            accuracy.append({"MobileNet": result})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
