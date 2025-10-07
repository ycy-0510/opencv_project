import cv2
import os

face_cascade = cv2.CascadeClassifier("face.xml")
print("Loaded Haar Cascade for face detection.")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Unable to open camera. Check device index and permissions.")

name = input("Enter your ID (e.g., 01, 02, ...): ")
if not name.isdigit() or len(name) != 2:
    raise ValueError("ID must be a two-digit number.")
print(f"Capturing images for ID: {name}")
# Save 16 images with face detected
count = 0
os.makedirs(f"face/face{name.zfill(2)}", exist_ok=True)
while count < 30:
    ret, photo = cap.read()
    if not ret:
        print("Failed to capture image")
        break
    gray = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.5, 5)
    for x, y, w, h in faces:
        cv2.imwrite(
            f"face/face{name.zfill(2)}/{count}.jpg",
            gray[
                y - 10 if y - 10 > 0 else 0 : (
                    y + h + 10 if y + h + 10 < gray.shape[0] else gray.shape[0]
                ),
                x - 10 if x - 10 > 0 else 0 : (
                    x + w + 10 if x + w + 10 < gray.shape[1] else gray.shape[1]
                ),
            ],
        )
        count += 1
        cv2.rectangle(photo, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(photo, f"Images Captured: {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if count >= 30:
            break
    cv2.imshow("Capturing Faces", photo)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    cv2.waitKey(20)  # wait 20ms between captures
print(f"Captured {count} images for ID: {name}")
cap.release()
cv2.destroyAllWindows()
