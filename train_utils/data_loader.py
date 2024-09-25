import os
import numpy as np
import cv2
from sklearn.utils import shuffle
from concurrent.futures import ThreadPoolExecutor
import config

def load_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (config.IMG_SIZE, config.IMG_SIZE))
    return img

def load_data():
    data = []
    labels = []
    
    for class_dir in os.listdir(config.DATA_DIR):
        class_path = os.path.join(config.DATA_DIR, class_dir)
        label = int(class_dir)

        with ThreadPoolExecutor() as executor:
            images = list(executor.map(load_image, [os.path.join(class_path, img) for img in os.listdir(class_path)]))

        data.extend(images)
        labels.extend([label] * len(images))

    data = np.array(data, dtype="float32") / 255.0  # Normalizando as imagens
    labels = np.array(labels)

    # Misturar os dados
    data, labels = shuffle(data, labels, random_state=42)

    return data, labels
