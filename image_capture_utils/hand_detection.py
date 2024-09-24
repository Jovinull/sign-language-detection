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

            # Adiciona um padding mínimo, mas remove as barras pretas grandes
            padding = 10  # Mantemos um padding pequeno
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(w, x_max + padding)
            y_max = min(h, y_max + padding)

            hand_crop = frame[y_min:y_max, x_min:x_max]
            hand_h, hand_w, _ = hand_crop.shape

            # Redimensionar para 512x512 sem distorcer a imagem
            if hand_h > hand_w:
                scale = IMG_SIZE / hand_h
            else:
                scale = IMG_SIZE / hand_w

            new_w = int(hand_w * scale)
            new_h = int(hand_h * scale)

            hand_resized = cv2.resize(hand_crop, (new_w, new_h))

            # Centralizar a imagem redimensionada em um fundo 512x512
            final_hand = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
            y_offset = (IMG_SIZE - new_h) // 2
            x_offset = (IMG_SIZE - new_w) // 2
            final_hand[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = hand_resized

            # Desenha o retângulo na imagem original para visualização
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            return final_hand, frame

    else:
        cv2.putText(frame, "Nenhuma mão detectada", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return None, frame
