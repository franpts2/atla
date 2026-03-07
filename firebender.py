import cv2

# initialize camera
cap = cv2.VideoCapture(0)

while True:
    # capture frame by frame
    success, frame = cap.read()
    if not success:
        break

    # flip the frame (like a mirror)
    frame = cv2.flip(frame,1)

    # get dimensions
    h, w, _ = frame.shape
    center_x, center_y = w // 2, h // 2

    # target in center
    # color is BGR (not RGB!)
    cv2.circle(frame, (center_x, center_y), 50, (0,255,0),2)

    # display resulting frame
    cv2.imshow("Bending Training", frame)

    # press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

