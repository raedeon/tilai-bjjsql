# v1: L model, 100 epochs, 960 img sz
# If OutOfMemoryError, run: nvidia-smi, then kill whichever is taking up the most space by: kill -9 <PID>
# v3 1280 img sz

import torch
torch.cuda.empty_cache()

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from ultralytics import YOLO

def main():
    # Check if GPU is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load the YOLOv8 model (large version)
    #model = YOLO("yolov8l6.pt")
    model = YOLO("runs/detect/v3_train_run2/weights/last.pt") # for training from last checkpoint

    # Train the model
    model.train(
        resume=True, # use when retraining a previous model e.g. v3
        data="data.yaml",   # path to data.yaml
        epochs=150,                         # number of training epochs
        imgsz=1280,                          # input image size
        batch=4,                           # batch size
        name="v3_train_run",                # run name
        device=device                       # auto-select device
    )

if __name__ == "__main__":
    main()