import cv2

recognizer = cv2.face.LBPHFaceRecognizer_create(
    radius=1, neighbors=8, grid_x=16, grid_y=16
)
recognizer.read("trainer.yml")
print("Loaded trained model from trainer.yml")
cascade_path = "face.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Unable to open camera. Check device index and permissions.")
print("Camera opened successfully.")
while True:
    ret, img = cap.read()
    if not ret:
        print("Failed to capture image")
        break
    img = cv2.resize(img, (600, 400))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.1,5)

    name = {
        0: "01",
        1: "03",
        2: "04",
        3: "05",
        4: "06",
        5: "07",
        6: "08",
        7: "09",
        8: "10",
        9: "11",
        10: "12",
        11: "13",
        12: "14",
        13: "15",
        14: "16",
        15: "17",
        16: "18",
        17: "19",
        18: "20",
        19: "21",
        20: "22",
        21: "23",
        22: "24",
        23: "25",
        24: "26",
        25: "27",
        26: "28",
        27: "29",
        28: "30",
        29: "31",
        30: "32",
        31: "33",
        32: "34",
        33: "mon",
    }
    for x, y, w, h in faces:
        idnum, confidence = recognizer.predict(gray[y : y + h, x : x + w])
        if confidence < 60:
            color = (0, 255, 0)
            text = f"{name[idnum]} {round(100 - confidence)}%"
        else:
            color = (0, 0, 255)
            text = "unknown"
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 5)
        cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("camera", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
