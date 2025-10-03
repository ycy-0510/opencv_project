import cv2
import numpy as np

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Unable to open camera. Check device index and permissions.")
print("Camera opened successfully.")
face_cascade = cv2.CascadeClassifier("face.xml")
print("Loaded Haar Cascade for face detection.")


# Gaussian blur the face
def blur_face_circle(frame, face_coords):
    # (x, y, w, h) = face_coords
    # face_region = frame[y:y+h, x:x+w]
    # blurred_face = cv2.GaussianBlur(face_region, (99, 99), 30)
    # frame[y:y+h, x:x+w] = blurred_face
    # return frame
    (x, y, w, h) = face_coords
    face_region = frame[y : y + h, x : x + w]
    mask = np.zeros(face_region.shape[:2], dtype="uint8")
    center = (w // 2, h // 2)
    radius = int(0.5 * (w + h) / 2)
    cv2.circle(mask, center, radius, 255, -1)
    blurred_face = cv2.GaussianBlur(
        face_region, (99, 99), borderType=cv2.BORDER_REPLICATE, sigmaX=30
    )
    face_region = np.where(mask[:, :, np.newaxis] == 255, blurred_face, face_region)
    frame[y : y + h, x : x + w] = face_region
    return frame


while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=2, minNeighbors=5)
    for x, y, w, h in faces:
        cv2.circle(frame, (x + w // 2, y + h // 2), (w + h) // 4, (0, 0, 255), 10)
        # frame = blur_face_circle(frame, (x, y, w, h))
    cv2.imshow("Face Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
