from ultralytics import YOLO

# Load pretrained YOLO model
model = YOLO("yolo11n.pt")

# Detect and track objects in the video
results = model.track(
    source="input_video.avi",
    save=True,
    tracker="bytetrack.yaml"
)

print("Object detection and tracking completed!")