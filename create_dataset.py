import os
import pickle
import cv2
import numpy as np

DATA_DIR = './data'
IMG_SIZE = 512  # O mesmo tamanho usado durante a captura

data = []
labels = []

# Percorre cada diretório (classe) no DATA_DIR
for dir_ in os.listdir(DATA_DIR):
    class_dir = os.path.join(DATA_DIR, dir_)
    
    if os.path.isdir(class_dir):
        # Para cada imagem dentro da classe
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            img = cv2.imread(img_path)

            if img is not None:
                # Redimensiona a imagem para o tamanho correto (caso necessário)
                img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

                # Normaliza os valores da imagem para a faixa [0, 1] e adiciona ao dataset
                img_resized = img_resized / 255.0
                data.append(img_resized)
                
                # Adiciona o rótulo (classe) correspondente
                labels.append(int(dir_))  # Presumindo que os nomes das pastas são números das classes

# Converte as listas para arrays NumPy
data = np.array(data)
labels = np.array(labels)

# Salva os dados em um arquivo pickle
os.makedirs('results', exist_ok=True)
with open('results/data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print(f"Dataset criado com {len(data)} exemplos.")
