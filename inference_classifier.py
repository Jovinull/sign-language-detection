import cv2
import numpy as np
import tensorflow as tf

# Carregar o modelo treinado
model = tf.keras.models.load_model('hand_gesture_cnn.h5')

# Dicionário de mapeamento de labels (Ajuste de acordo com o número de classes)
labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'I', 8: 'L', 9: 'M', 
               10: 'N', 11: 'O', 12: 'P', 13: 'Q', 14: 'R', 15: 'S', 16: 'T', 17: 'U', 18: 'V', 
               19: 'W', 20: 'NENHUM SINAL'}

cap = cv2.VideoCapture(0)

IMG_SIZE = 128  # Mesmo tamanho usado no treinamento

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Redimensiona e normaliza a imagem
    resized_frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    normalized_frame = np.array(resized_frame, dtype="float32") / 255.0
    normalized_frame = np.expand_dims(normalized_frame, axis=0)

    # Faz a predição
    prediction = model.predict(normalized_frame)
    predicted_class = np.argmax(prediction)

    # Exibe o resultado
    label = labels_dict[predicted_class]
    cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Hand Gesture Classification', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
