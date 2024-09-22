import os
import numpy as np
import cv2
import tensorflow as tf
from keras import layers, models
from sklearn.model_selection import train_test_split

# Carregando e processando os dados
DATA_DIR = './data'
IMG_SIZE = 128  # Tamanho das imagens

def load_data():
    data = []
    labels = []
    
    for class_dir in os.listdir(DATA_DIR):
        class_path = os.path.join(DATA_DIR, class_dir)
        label = int(class_dir)
        
        for img_path in os.listdir(class_path):
            img = cv2.imread(os.path.join(class_path, img_path))
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            data.append(img)
            labels.append(label)
    
    data = np.array(data, dtype="float32") / 255.0  # Normalizando as imagens
    labels = np.array(labels)
    
    return data, labels

data, labels = load_data()

# Dividindo os dados em conjunto de treino e teste
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Criando o modelo CNN do zero
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(21, activation='softmax')  # Ajustar o número de classes
])

# Compilando o modelo
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Treinando o modelo
history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# Salvando o modelo treinado
model.save('hand_gesture_cnn.h5')
