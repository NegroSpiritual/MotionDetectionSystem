from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import cv2
import os
import numpy
import uvicorn
import subprocess

app = FastAPI()

# Templates configuration
templates = Jinja2Templates(directory="templates")

# Create a folder for the dataset
dataset_folder = 'dataset'
if not os.path.exists(dataset_folder):
    os.makedirs(dataset_folder)

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Number of pictures to take
num_pictures = 10

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/capture")
async def capture_image(label: str = Form(...)):
    for i in range(num_pictures):
        # Capture frame from the webcam
        ret, frame = cap.read()

        # Save the image with the label
        image_filename = f"{label}_{i+1}.jpg"
        image_path = os.path.join(dataset_folder, image_filename)
        cv2.imwrite(image_path, frame)

        print(f"Image {i+1} captured and labeled as '{label}'.")
        cv2.waitKey(500)

    return {"message": "Image capture completed. Check the 'dataset' folder for the labeled images."}

    if __name__ == "__main__":
        uvicorn_command = "uvicorn main:app --reload"
        subprocess.run(uvicorn_command, shell=True)