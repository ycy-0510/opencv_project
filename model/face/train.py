import os
import cv2
import numpy as np
import cv2

detector = cv2.CascadeClassifier("face.xml")
print("Loaded Haar Cascade for face detection.")
recog = cv2.face.LBPHFaceRecognizer_create()
print("Initialized LBPH Face Recognizer.")
faces = []
ids = []


folders = os.listdir("face")
folders.sort()
folders = [folder for folder in folders if folder.startswith("face")]
print(folders)

for folder, idx in zip(folders, range(len(folders))):
    # list files
    print(f"Processing folder: face/{folder} with ID: {idx}")
    for file in os.listdir(f"face/{folder}"):
        if file.endswith(".jpg") or file.endswith(".png"):
            img = cv2.imread(f"face/{folder}/{file}")
            img = cv2.resize(img, (300, 400))
            grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_np = np.array(grey, "uint8")
            face = detector.detectMultiScale(grey,1.1,4)
            for x, y, w, h in face:
                cv2.rectangle(img_np, (x, y), (x + w, y + h), (0, 255, 0), 2)
                faces.append(img_np[y : y + h, x : x + w])
                cv2.imshow("training", img_np[y : y + h, x : x + w])
                ids.append(idx)
            cv2.imshow("training", img_np)
            cv2.waitKey(5)
print("training...")
recog.train(faces, np.array(ids))
recog.save("trainer.yml")
print("training complete")
