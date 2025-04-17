import os
import cv2
import mediapipe as mp
import tensorflow as tf
import numpy as np
from collections import deque


os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


norm_file = "norm_params.npz"
if os.path.exists(norm_file):
    params = np.load(norm_file)
    mean = params["mean"].astype(np.float32)
    std = params["std"].astype(np.float32)
    print("Normalization parameters loaded.")
else:
    print("Normalization parameters not found.")
    mean, std = None, None


labels_file = "gesture_labels.txt"
if os.path.exists(labels_file):
    with open(labels_file, "r") as f:
        gesture_labels = [line.strip() for line in f if line.strip()]
    print("Gesture labels loaded:", gesture_labels)
else:
    gesture_labels = ["hello", "ok", "peace"]
    print("Using fallback gesture labels:", gesture_labels)


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils


model_path = "gesture_model.tflite"
if not os.path.exists(model_path):
    raise FileNotFoundError("TFLite model file not found.")
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("TFLite model loaded, input shape:", input_details[0]['shape'])

DEBUG = True


def preprocess_keypoints(hand_list):

    combined = []
    if not hand_list:
        combined = [0.0] * 126
    else:
        if len(hand_list) >= 2:

            sorted_hands = sorted(hand_list, key=lambda h: np.mean([lm.x for lm in h.landmark]))
            for h in sorted_hands[:2]:
                kp = []
                for lm in h.landmark:
                    kp.extend([lm.x, lm.y, lm.z])
                combined.extend(kp)
        else:
            h = hand_list[0]
            kp = []
            for lm in h.landmark:
                kp.extend([lm.x, lm.y, lm.z])
            combined.extend(kp)
            combined.extend([0.0] * 63)
    if len(combined) != 126:
        combined += [0.0] * (126 - len(combined))
    arr = np.array(combined, dtype=np.float32)
    if mean is not None and std is not None:
        arr = (arr - mean) / std
    return arr.reshape(1, -1)


def classify_gesture(hand_list):
    inp = preprocess_keypoints(hand_list)
    interpreter.set_tensor(input_details[0]['index'], inp)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])[0]  # shape: (num_classes,)
    if DEBUG:
        print("Raw output:", output)
    idx = int(np.argmax(output))
    conf = output[idx]
    threshold = 0.6
    if conf < threshold:
        return "Unknown", conf, output
    return gesture_labels[idx], conf, output


SMOOTHING_WINDOW = 5
pred_queue = deque(maxlen=SMOOTHING_WINDOW)


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam.")

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    frame = cv2.flip(frame, 1)  # Mirror view
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    hand_list = []
    if results.multi_hand_landmarks:
        for h in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, h, mp_hands.HAND_CONNECTIONS)
            hand_list.append(h)

    if hand_list:
        gesture, conf, raw_vals = classify_gesture(hand_list)
        pred_queue.append((gesture, conf))
        # For smoothing: use majority vote from the queue.
        gestures = [g for g, c in pred_queue]
        smoothed = max(set(gestures), key=gestures.count)
        display_text = f"{smoothed} ({conf * 100:.1f}%)"
        cv2.putText(frame, display_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2, cv2.LINE_AA)
    else:
        pred_queue.clear()

    cv2.imshow("Real-Time Gesture Recognition", frame)
    if cv2.waitKey(5) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
