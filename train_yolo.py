from ultralytics import YOLO

# Load a pre-trained model
model = YOLO("yolo11n.pt")  # Load a pre-trained YOLOv11 model, replace with the correct model version

# Train the model using the specified YAML file
results = model.train(data="dataset/Fruit_Vege_Recognition/data.yaml", epochs=50, imgsz=640)

# Print out the results
print("Training completed. The model results are saved in: ", results)
