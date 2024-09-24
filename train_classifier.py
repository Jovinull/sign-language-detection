import os
import numpy as np
import cv2
import tensorflow as tf
from keras import layers, models
from sklearn.model_selection import KFold
from sklearn.utils import shuffle

# Carregando e processando os dados
DATA_DIR = './data'
IMG_SIZE = 128  # Tamanho das imagens
N_SPLITS = 10  # Número de divisões para a validação cruzada

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

# Misturar os dados antes da validação cruzada
data, labels = shuffle(data, labels, random_state=42)

# Definir o KFold para validação cruzada
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

# Criando o modelo CNN do zero
def create_model():
    model = models.Sequential([
        layers.Conv2D(16, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),  # Filtros reduzidos para 16
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(32, (3, 3), activation='relu'),  # Filtros reduzidos para 32
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu'),  # Redução para 64 na última convolucional
        layers.MaxPooling2D((2, 2)),
        
        layers.Flatten(),
        layers.Dense(128, activation='relu'),  # Redução para 128 neurônios
        layers.Dropout(0.5),  # Adicionado dropout para evitar overfitting
        layers.Dense(21, activation='softmax')
    ])
    
    # Compilar o modelo
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Executar validação cruzada
fold_no = 1
for train_index, test_index in kf.split(data):
    x_train, x_test = data[train_index], data[test_index]
    y_train, y_test = labels[train_index], labels[test_index]

    # Criar um novo modelo para cada iteração
    model = create_model()

    print(f'Treinando a fold {fold_no}...')

    # Treinar o modelo
    history = model.fit(x_train, y_train, epochs=25, validation_data=(x_test, y_test))
    
    # Avaliar a acurácia no conjunto de teste
    scores = model.evaluate(x_test, y_test, verbose=0)
    print(f'Acurácia da fold {fold_no}: {scores[1] * 100}%')
    
    fold_no += 1

# Salvando o último modelo treinado
model.save('hand_gesture_cnn_kfold.h5')
