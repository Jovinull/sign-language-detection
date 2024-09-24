import cv2
import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

def process_frame(frame, IMG_SIZE):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            h, w, c = frame.shape
            x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * w)
            y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * h)
            x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * w)
            y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * h)

            padding = 30
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(w, x_max + padding)
            y_max = min(h, y_max + padding)

            hand_crop = frame[y_min:y_max, x_min:x_max]
            hand_h, hand_w, _ = hand_crop.shape

            if hand_h != hand_w:
                max_dim = max(hand_h, hand_w)
                square_hand = np.zeros((max_dim, max_dim, 3), dtype=np.uint8)
                if hand_h > hand_w:
                    offset = (hand_h - hand_w) // 2
                    square_hand[:, offset:offset + hand_w] = hand_crop
                else:
                    offset = (hand_w - hand_h) // 2
                    square_hand[offset:offset + hand_h, :] = hand_crop
                hand_crop = square_hand

            hand_resized = cv2.resize(hand_crop, (IMG_SIZE, IMG_SIZE))
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            return hand_resized, frame

    else:
        cv2.putText(frame, "Nenhuma mao detectada", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return None, frame
