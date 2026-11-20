import cv2

cap = cv2.VideoCapture("rtsp://localhost:8554/drone")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    cv2.imshow("DJI Feed", frame)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()