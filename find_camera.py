import cv2

for camera_id in range(0, 20):
    cap = cv2.VideoCapture(camera_id)
    if cap is None or not cap.isOpened():
        print(f"No camera found at index {camera_id}")
    else:
        ret, frame = cap.read()
        if ret:
            print(f"camera found at index {camera_id}")
            cv2.imshow('frame', frame)
            cv2.waitKey(0)
            cv2.destroyWindow('frame')
        cap.release()

cv2.destroyWindow