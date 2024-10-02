import h5py
import numpy as np
from sklearn.utils import shuffle
import config
import tensorflow as tf

def load_data_hdf5():
    """
    Carrega os dados e rótulos a partir do arquivo HDF5.
    
    Retorna:
        h5f (h5py.File): Objeto do arquivo HDF5 aberto.
        labels (numpy.ndarray): Array de rótulos.
    """
    HDF5_FILE = 'results/data.h5'
    h5f = h5py.File(HDF5_FILE, 'r')
    labels = h5f['labels'][:]  # Carrega os rótulos aqui
    return h5f, labels

def load_data_hdf5_as_dataset(h5f, labels, indices):
    """
    Cria um TensorFlow Dataset a partir dos dados no arquivo HDF5 para os índices fornecidos.

    Args:
        h5f (h5py.File): Objeto do arquivo HDF5 aberto.
        labels (numpy.ndarray): Array de rótulos.
        indices (numpy.ndarray): Índices dos dados a serem carregados.

    Returns:
        tf.data.Dataset: Dataset contendo os dados correspondentes aos índices.
    """
    data_shape = (config.IMG_SIZE, config.IMG_SIZE, 3)  # Atualize se necessário

    def generator():
        for i in indices:
            yield h5f['data'][i], labels[i]

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=data_shape, dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32)
        )
    )
    
    # Remover o mapeamento de redimensionamento
    return dataset
