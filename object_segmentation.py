import cv2
import os
import numpy as np
from ultralytics import YOLO

model = YOLO("yolov8n-seg.pt")

frame_folder = "frames"
output_folder = "objects"

os.makedirs(output_folder, exist_ok=True)

count = 0

for file in os.listdir(frame_folder):

    path = os.path.join(frame_folder, file)
    img = cv2.imread(path)

    if img is None:
        continue

    h, w = img.shape[:2]

    results = model(img)

    for r in results:

        if r.masks is None:
            continue

        masks = r.masks.data.cpu().numpy()

        for mask in masks:

            # resize mask to image size
            mask = cv2.resize(mask, (w, h))

            mask = (mask > 0.5).astype("uint8") * 255

            object_img = cv2.bitwise_and(img, img, mask=mask)

            save_path = f"{output_folder}/object_{count}.png"
            cv2.imwrite(save_path, object_img)

            count += 1

print("Objects extracted:", count)