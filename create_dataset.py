import os
import cv2
import numpy as np
import h5py
import multiprocessing as mp
from functools import partial

# Importar parâmetros do config.py
import config

DATA_DIR = config.DATA_DIR  # Diretório contendo os dados organizados por classes
IMG_SIZE = config.IMG_SIZE  # Tamanho das imagens fixo
HDF5_FILE = 'results/data.h5'  # Caminho para o arquivo HDF5
BATCH_SIZE = 1000  # Número de exemplos a serem processados por vez

# Mapeamento de métodos de interpolação
INTERPOLATION_METHODS = {
    'INTER_AREA': cv2.INTER_AREA,
    'INTER_LINEAR': cv2.INTER_LINEAR,
    'INTER_CUBIC': cv2.INTER_CUBIC
}

def count_data_and_classes(data_dir):
    total = 0
    classes = set()
    for dir_ in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, dir_)
        if os.path.isdir(class_dir):
            try:
                classes.add(int(dir_))
                total += len([name for name in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, name))])
            except ValueError:
                print(f"Pasta {dir_} não representa uma classe válida (deve ser um número). Ignorando.")
    return total, sorted(list(classes))

def process_image(img_path, img_size, interpolation):
    img = cv2.imread(img_path)
    if img is not None:
        try:
            img_resized = cv2.resize(img, (img_size, img_size), interpolation=interpolation)
            img_normalized = img_resized.astype(np.float32) / 255.0
            return img_normalized
        except Exception as e:
            print(f"Erro ao redimensionar {img_path}: {e}")
            return None
    else:
        print(f"Erro ao ler a imagem {img_path}.")
        return None

def create_hdf5_dataset_parallel(data_dir, hdf5_file, img_size, batch_size, interpolation=cv2.INTER_AREA):
    total_samples, classes = count_data_and_classes(data_dir)
    num_classes = len(classes)

    with h5py.File(hdf5_file, 'w') as h5f:
        # Cria datasets redimensionáveis
        h5f.create_dataset('data', shape=(0, img_size, img_size, 3), maxshape=(None, img_size, img_size, 3), dtype=np.float32)
        h5f.create_dataset('labels', shape=(0,), maxshape=(None,), dtype=np.int32)

        data_buffer = []
        labels_buffer = []
        count = 0

        # Preparar lista de caminhos e rótulos
        paths_labels = []
        for dir_ in os.listdir(data_dir):
            class_dir = os.path.join(data_dir, dir_)
            if os.path.isdir(class_dir):
                try:
                    class_label = int(dir_)  # Presumindo que o nome da pasta é o rótulo da classe
                except ValueError:
                    print(f"Pasta {dir_} não representa uma classe válida (deve ser um número). Ignorando.")
                    continue

                for img_name in os.listdir(class_dir):
                    img_path = os.path.join(class_dir, img_name)
                    if not os.path.isfile(img_path):
                        continue
                    paths_labels.append((img_path, class_label))

        # Função parcial para passar argumentos fixos
        func = partial(process_image, img_size=img_size, interpolation=interpolation)

        # Utilizar multiprocessing para processar as imagens em paralelo
        with mp.Pool(mp.cpu_count()) as pool:
            for i, img_normalized in enumerate(pool.imap(func, [pl[0] for pl in paths_labels]), 1):
                if img_normalized is not None:
                    data_buffer.append(img_normalized)
                    labels_buffer.append(paths_labels[i-1][1])
                    count += 1

                    if len(data_buffer) >= batch_size:
                        # Redimensiona os datasets para acomodar os novos dados
                        h5f['data'].resize((count, img_size, img_size, 3))
                        h5f['labels'].resize((count,))

                        # Escreve os dados no arquivo HDF5
                        h5f['data'][-batch_size:] = np.array(data_buffer, dtype=np.float32)
                        h5f['labels'][-batch_size:] = np.array(labels_buffer, dtype=np.int32)

                        # Limpa os buffers
                        data_buffer = []
                        labels_buffer = []

                        print(f"{count}/{total_samples} imagens processadas.")

        # Escreve quaisquer dados restantes no buffer
        if data_buffer:
            h5f['data'].resize((count, img_size, img_size, 3))
            h5f['labels'].resize((count,))
            h5f['data'][-len(data_buffer):] = np.array(data_buffer, dtype=np.float32)
            h5f['labels'][-len(labels_buffer):] = np.array(labels_buffer, dtype=np.int32)
            print(f"{count}/{total_samples} imagens processadas.")

    print(f"Arquivo HDF5 criado com {count} exemplos em {hdf5_file}.")

if __name__ == "__main__":
    os.makedirs('results', exist_ok=True)
    interpolation_method = config.INTERPOLATION_METHOD
    if interpolation_method not in INTERPOLATION_METHODS:
        raise ValueError(f"Método de interpolação inválido: {interpolation_method}. "
                         f"Escolha entre {list(INTERPOLATION_METHODS.keys())}.")
    create_hdf5_dataset_parallel(
        DATA_DIR,
        HDF5_FILE,
        IMG_SIZE,
        BATCH_SIZE,
        interpolation=INTERPOLATION_METHODS[interpolation_method]
    )
