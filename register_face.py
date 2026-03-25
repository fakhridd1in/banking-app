import cv2
import numpy as np
import os
import time

save_dir = "my_face"
os.makedirs(save_dir, exist_ok=True)

cap = cv2.VideoCapture(0)
count = 0
total = 20

print(f"We will take {total} photos of your face.")
print("Look at the camera. Press SPACE to capture each photo.")

while count < total:
    ret, frame = cap.read()
    if not ret:
        break

    display = frame.copy()
    cv2.putText(display, f"Photos taken: {count}/{total}", (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(display, "Press SPACE to capture", (30, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.imshow("Register Your Face", display)

    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):
        path = f"{save_dir}/face_{count}.jpg"
        cv2.imwrite(path, frame)
        count += 1
        print(f"Captured {count}/{total}")

cap.release()
cv2.destroyAllWindows()
print("Registration complete. Run login.py next.")