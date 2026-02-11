from ultralytics import YOLO
import os

def main():

    project_root = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(project_root, "transparent.yaml")

    print("Using YAML:", yaml_path)

    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"YAML file not found at: {yaml_path}")

    #Load the Yolo
    model = YOLO("yolov8n-seg.pt")

    #Train it
    model.train(
        data="transparent.yaml",
        epochs=100,
        imgsz=640,
        batch=4,
        device="cpu",  #CPU only, GPU don't work cause our system ain't compatible :\
        task="segment"
    )


if __name__ == "__main__":
    main()
