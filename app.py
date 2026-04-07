import cv2
import mediapipe as mp
import numpy as np
import time
import serial

# ---------------- SERIAL ----------------
SERIAL_PORT = "COM5"
BAUD_RATE = 9600

try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    print("Connected to Arduino:", SERIAL_PORT)
    time.sleep(2)
except:
    ser = None
    print("Serial not connected")

# ---------------- EAR ----------------
def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def compute_ear(landmarks, w, h):
    def get_xy(i):
        return landmarks[i].x * w, landmarks[i].y * h

    left = (euclidean(get_xy(159), get_xy(145)) +
            euclidean(get_xy(158), get_xy(153))) / \
           (2 * euclidean(get_xy(33), get_xy(133)))

    right = (euclidean(get_xy(386), get_xy(374)) +
             euclidean(get_xy(385), get_xy(380))) / \
            (2 * euclidean(get_xy(263), get_xy(362)))

    return (left + right) / 2

# ---------------- MAIN ----------------
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

EAR_THRESHOLD = 0.25

eye_closed_start = None
last_sent = ""
drowsy_locked = False   # 🔥 IMPORTANT

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb)
    h, w = frame.shape[:2]

    label = "NEUTRAL"
    ear_value = 0

    if results.multi_face_landmarks:
        face = results.multi_face_landmarks[0]

        ear_value = compute_ear(face.landmark, w, h)

        if ear_value < EAR_THRESHOLD:
            if eye_closed_start is None:
                eye_closed_start = time.time()

            duration = time.time() - eye_closed_start

            if duration >= 3:
                label = "DROWSY"
            elif duration >= 1:
                label = "SLEEPY"
        else:
            eye_closed_start = None
            label = "NEUTRAL"

    # 🔥 LOCK SYSTEM
    if label == "DROWSY":
        drowsy_locked = True

    if drowsy_locked:
        label = "DROWSY"

    # ---------------- SERIAL SEND ----------------
    if ser and label != last_sent:
        try:
            ser.write((label + "\n").encode())
            print("Sent:", label)
            last_sent = label
        except:
            print("Serial error")

    # ---------------- DISPLAY ----------------
    color = (0, 255, 0)
    if label == "SLEEPY":
        color = (0, 165, 255)
    elif label == "DROWSY":
        color = (0, 0, 255)

    cv2.putText(frame, label, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.putText(frame, f"EAR: {ear_value:.2f}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    cv2.imshow("Driver Monitoring", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

if ser:
    ser.close()