from ultralytics import YOLO

## Link to check all yolo models - https://github.com/ultralytics/ultralytics
## We choose v8 for now (smaller model => faster calculations => accuracy might be lesser compared to bigger models)
model = YOLO('yolov8m')

## Save the results of predeiction
results = model.predict('input_videos/08fd33_4.mp4', save=True)
print(results[0])
print("================================")
for box in results[0].boxes:
    print(box)