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

def get_train_test_split(h5f, labels, train_indices, test_indices):
    """
    Obtém os dados de treino e teste a partir do arquivo HDF5 usando índices.

    Args:
        h5f (h5py.File): Objeto do arquivo HDF5 aberto.
        labels (numpy.ndarray): Array de rótulos.
        train_indices (numpy.ndarray): Índices para o conjunto de treino.
        test_indices (numpy.ndarray): Índices para o conjunto de teste.

    Returns:
        x_train (numpy.ndarray): Dados de treino.
        y_train (numpy.ndarray): Rótulos de treino.
        x_test (numpy.ndarray): Dados de teste.
        y_test (numpy.ndarray): Rótulos de teste.
    """
    x_train = h5f['data'][train_indices]
    y_train = labels[train_indices]
    x_test = h5f['data'][test_indices]
    y_test = labels[test_indices]
    
    return x_train, y_train, x_test, y_test

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
    def generator():
        for i in indices:
            yield h5f['data'][i], labels[i]  # Retorne os dados e os rótulos

    return tf.data.Dataset.from_generator(generator,
                                          output_signature=(
                                              tf.TensorSpec(shape=(512, 512, 3), dtype=tf.float32),
                                              tf.TensorSpec(shape=(), dtype=tf.int32)
                                          ))
