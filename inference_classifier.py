import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp

# Inicializa o Mediapipe para detecção de mãos
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)  # Detecção de apenas 1 mão

# Carregar o modelo treinado
model = tf.keras.models.load_model('results/hand_gesture_cnn_kfold.h5')

# Dicionário de mapeamento de labels
labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'I', 8: 'L', 9: 'M', 
               10: 'N', 11: 'O', 12: 'P', 13: 'Q', 14: 'R', 15: 'S', 16: 'T', 17: 'U', 18: 'V', 
               19: 'W'}

cap = cv2.VideoCapture(0)

IMG_SIZE = 512  # Mesmo tamanho usado no treinamento

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Converte o frame para RGB (necessário para o Mediapipe)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detecta mãos no frame
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Obter as coordenadas da mão para desenhar o quadrado ao redor
            h, w, _ = frame.shape
            x_min, y_min = w, h
            x_max, y_max = 0, 0
            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                x_min, y_min = min(x_min, x), min(y_min, y)
                x_max, y_max = max(x_max, x), max(y_max, y)

            # Recortar a região da mão
            hand_roi = frame[y_min:y_max, x_min:x_max]

            if hand_roi.size > 0:
                # Redimensiona a imagem da mão diretamente para 512x512
                hand_resized = cv2.resize(hand_roi, (IMG_SIZE, IMG_SIZE))

                # Normaliza a imagem
                normalized_hand = np.array(hand_resized, dtype="float32") / 255.0
                normalized_hand = np.expand_dims(normalized_hand, axis=0)

                # Faz a predição
                prediction = model.predict(normalized_hand)
                predicted_class = np.argmax(prediction)

                # Exibe o resultado
                label = labels_dict[predicted_class]
                cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Desenha um quadrado verde ao redor da mão detectada
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    # Exibe o frame com a detecção e classificação
    cv2.imshow('Hand Gesture Classification', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
