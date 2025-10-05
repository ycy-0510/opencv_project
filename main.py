import cv2
import numpy as np
from imutils.perspective import four_point_transform

cap = cv2.VideoCapture(1)

WIDTH, HEIGHT = 800, 600
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)  # set width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)  # set height

if not cap.isOpened():
    raise RuntimeError("Unable to open camera. Check device index and permissions.")

def focus_measure(image):
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(grey, cv2.CV_64F)
    fm = lap.var()
    return fm

def scan_detection(image):
    global document_contour
    document_contour = np.array([[0, 0], [WIDTH, 0], [WIDTH, HEIGHT], [0, HEIGHT]])
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey, (5, 5), 0)
    _, threshold = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    max_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:
            peri = cv2.arcLength(contour, True)
            aprox = cv2.approxPolyDP(contour, 0.015 * peri, True)
            if area > max_area and len(aprox) == 4:
                max_area = area
                document_contour = aprox
    cv2.drawContours(image, [document_contour], -1, (0, 255, 0), 3)

prev_wrap = None
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_copy = frame.copy()
    scan_detection(frame_copy)

    cv2.imshow("input", frame_copy)
    wrapped = four_point_transform(frame, document_contour.reshape(4, 2))
    current_wrap = cv2.resize(wrapped, (80, 60))
    diff = cv2.absdiff(cv2.cvtColor(prev_wrap, cv2.COLOR_BGR2GRAY), cv2.cvtColor(current_wrap, cv2.COLOR_BGR2GRAY)) if prev_wrap is not None else None
    # Count white area in wrapped image (cvt to grey and value > 150)
    white_area = cv2.inRange(cv2.cvtColor(wrapped, cv2.COLOR_BGR2GRAY), 150, 255)
    white_area_in_wrapped_rate = cv2.countNonZero(white_area) / (white_area.shape[0] * white_area.shape[1])
    cv2.imshow("white area", white_area)
    if diff is not None:
        cv2.imshow("diff", diff)
        non_zero_count = cv2.countNonZero(diff)
        if non_zero_count < 5000:
            print("No significant change detected, skipping update.")
            if white_area_in_wrapped_rate > 0.5:
                if focus_measure(wrapped) < 100:
                    print("Image is blurry, skipping update.")
                    continue
                print("Document detected and stable, saving scanned image.")
                cv2.imwrite("scan/scanned.jpg", wrapped)
                exit(0)
            else:
                print(f"No document detected. {white_area_in_wrapped_rate * 100:.2f}% white area.")
        else:
            print(non_zero_count)
    cv2.imshow("scanned", wrapped)
    prev_wrap = cv2.resize(wrapped, (80, 60))
    if cv2.waitKey(50) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
