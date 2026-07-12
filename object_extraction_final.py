import cv2
import numpy as np

video_path = "input_video.mp4"

cap = cv2.VideoCapture(video_path)

drawing = False
points = []
paused = False
tracking = False
tracker = None
mask = None
init_bbox = None

scale = 0.7   # reduce size (0.5 smaller, 1 = original)

def draw(event, x, y, flags, param):
    global drawing, points

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        points = [(x, y)]

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            points.append((x, y))

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False


cv2.namedWindow("Input Video")
cv2.setMouseCallback("Input Video", draw)

while True:

    if not paused:
        ret, frame = cap.read()
        if not ret:
            break

        # resize frame 
        frame = cv2.resize(frame, None, fx=scale, fy=scale)

    display = frame.copy()

    if paused:
        for i in range(1, len(points)):
            cv2.line(display, points[i-1], points[i], (0,255,0), 2)

    cv2.imshow("Input Video", display)

    key = cv2.waitKey(30) & 0xFF

    if key == ord('p'):
        paused = True
        print("Paused - Draw object")

    if key == 13 and paused and len(points) > 5:

        pts = np.array(points, np.int32)

        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)

        x,y,w,h = cv2.boundingRect(pts)
        init_bbox = (x,y,w,h)

        tracker = cv2.TrackerCSRT_create()
        tracker.init(frame, init_bbox)

        tracking = True
        paused = False
        print("Tracking started")

    if tracking:

        success, bbox = tracker.update(frame)

        if success:
            x,y,w,h = [int(v) for v in bbox]

            #  shift mask based on movement
            dx = x - init_bbox[0]
            dy = y - init_bbox[1]

            shifted_mask = np.zeros_like(mask)

            M = np.float32([[1, 0, dx], [0, 1, dy]])
            shifted_mask = cv2.warpAffine(mask, M, (mask.shape[1], mask.shape[0]))

            result = cv2.bitwise_and(frame, frame, mask=shifted_mask)

            cv2.imshow("Tracked Object", result)

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()