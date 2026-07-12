import cv2
import os

video_path = "input_video.mp4"
output_folder = "frames"

os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(video_path)

count = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    filename = f"{output_folder}/frame_{count}.jpg"
    cv2.imwrite(filename, frame)

    count += 1

print("Frames extracted:", count)

cap.release()