import os
import numpy as np
import h5py
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
from scikeras.wrappers import KerasClassifier
from keras import layers, models, regularizers
from tensorflow.python.keras.utils.data_utils import Sequence
import config

# Caminho para o arquivo HDF5 criado anteriormente
HDF5_FILE = 'results/data.h5'

class HDF5DataGenerator(Sequence):
    """
    Gerador de dados para carregar dados em lotes a partir de um arquivo HDF5 em pedaços menores.
    """
    def __init__(self, hdf5_file, batch_size):
        self.hdf5_file = hdf5_file
        self.batch_size = batch_size
        with h5py.File(hdf5_file, 'r') as h5f:
            self.num_samples = h5f['data'].shape[0]  # Número total de amostras no conjunto de dados

    def __len__(self):
        return int(np.ceil(self.num_samples / self.batch_size))  # Número de batches

    def __getitem__(self, idx):
        """
        Carrega um batch de dados do arquivo HDF5.
        """
        # Abrir o arquivo HDF5
        with h5py.File(self.hdf5_file, 'r') as h5f:
            data = h5f['data']
            labels = h5f['labels']

            # Definir os índices de início e fim do batch
            start = idx * self.batch_size
            end = min((idx + 1) * self.batch_size, self.num_samples)

            # Ler o batch de dados e labels
            batch_data = data[start:end]
            batch_labels = labels[start:end]

        # Retornar o batch de dados e labels
        return batch_data, batch_labels

def create_model(conv_layers, filters, kernel_size, dense_units, dropout_rate, activation, optimizer, l2_reg, img_size):
    """
    Cria e compila o modelo Keras com base nos hiperparâmetros fornecidos.
    """
    if isinstance(filters, int):
        filters = [filters] * conv_layers  # Cria uma lista com o mesmo valor para cada camada

    model = models.Sequential()

    # Camadas convolucionais com Batch Normalization e MaxPooling
    for i in range(conv_layers):
        filters_layer = filters[i]
        if i == 0:
            model.add(layers.Conv2D(filters_layer, (kernel_size, kernel_size), activation=activation,
                                    input_shape=(img_size, img_size, 3),
                                    kernel_regularizer=regularizers.l2(l2_reg)))
        else:
            model.add(layers.Conv2D(filters_layer, (kernel_size, kernel_size), activation=activation,
                                    kernel_regularizer=regularizers.l2(l2_reg)))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2, 2)))

    # Global Average Pooling
    model.add(layers.GlobalAveragePooling2D())

    # Camada Densa e Saída
    model.add(layers.Dense(dense_units, activation=activation,
                           kernel_regularizer=regularizers.l2(l2_reg)))
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(20, activation='softmax'))  # Ajuste o número de unidades de saída conforme necessário

    # Compilando o Modelo
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

def choose_search_method(search_type, n_iter, param_grid, model, cv_splits):
    """
    Define o método de busca de hiperparâmetros (GridSearch ou RandomizedSearch).
    """
    scoring = {
        'accuracy': 'accuracy',
        'precision_weighted': make_scorer(precision_score, average='weighted'),
        'recall_weighted': make_scorer(recall_score, average='weighted'),
        'f1_weighted': make_scorer(f1_score, average='weighted')
    }

    if search_type == 'grid':
        print("Usando GridSearchCV...")
        search = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv_splits, 
                              verbose=2, scoring=scoring, refit='f1_weighted', n_jobs=-1)
    elif search_type == 'random':
        print(f"Usando RandomizedSearchCV com {n_iter} iterações...")
        search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=n_iter, 
                                    cv=cv_splits, verbose=2, random_state=42, scoring=scoring, 
                                    refit='f1_weighted', n_jobs=-1)
    else:
        raise ValueError("search_type deve ser 'grid' ou 'random'.")

    return search

def save_results_to_txt(search_result, filename="results/tuning_results.txt"):
    """
    Salva os resultados do tuning em um arquivo de texto.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, "w") as file:
        file.write("Melhores parâmetros:\n")
        file.write(str(search_result.best_params_) + "\n\n")
        
        file.write(f"Melhor F1 ponderado: {search_result.best_score_ * 100:.2f}%\n\n")
        
        for i, candidate in enumerate(search_result.cv_results_['params']):
            file.write(f"Configuração {i+1}: {candidate}\n")
            file.write(f"Acurácia média: {search_result.cv_results_['mean_test_accuracy'][i]:.4f}\n")
            file.write(f"F1-Score ponderado: {search_result.cv_results_['mean_test_f1_weighted'][i]:.4f}\n")
            file.write(f"Desvio padrão do F1: {search_result.cv_results_['std_test_f1_weighted'][i]:.4f}\n\n")

def main():
    # Definir o modelo KerasClassifier com scikeras
    model = KerasClassifier(
        model=create_model,
        conv_layers=config.conv_layers,
        filters=config.filters,
        kernel_size=config.kernel_size,
        dense_units=config.dense_units,
        dropout_rate=config.dropout_rate,
        activation=config.activation,
        optimizer=config.optimizer,
        epochs=config.epochs,
        batch_size=config.BATCH_SIZE,  # Valor inicial, será ajustado durante o tuning
        img_size=config.IMG_SIZE,  # Passar o img_size fixo
        verbose=1
    )

    # Definir os parâmetros de tuning a partir do config.py
    param_grid = {
        'model__conv_layers': config.param_grid['conv_layers'],
        'model__filters': config.param_grid['filters'],
        'model__kernel_size': config.param_grid['kernel_size'],
        'model__dense_units': config.param_grid['dense_units'],
        'model__dropout_rate': config.param_grid['dropout_rate'],
        'model__activation': config.param_grid['activation'],
        'model__optimizer': config.param_grid['optimizer'],
        'model__l2_reg': config.param_grid['l2_reg'],
        'epochs': config.param_grid['epochs'],
        'batch_size': config.param_grid['batch_size']
    }

    # Escolher o método de busca (Grid ou Random) a partir do config.py
    search_type = config.search_type  # 'grid' ou 'random'
    n_iter = config.n_iter            # Número de iterações para RandomizedSearchCV
    cv_splits = config.N_SPLITS       # Número de folds para cross-validation

    # Definir o método de busca
    search_method = choose_search_method(search_type, n_iter, param_grid, model, cv_splits)

    # Definir o gerador de dados
    batch_size = config.BATCH_SIZE
    data_generator = HDF5DataGenerator(HDF5_FILE, batch_size)

    # Preparar os dados (X e y) a partir do gerador
    # Como scikeras suporta o uso de geradores, podemos passar o data_generator diretamente
    # Porém, para GridSearchCV e RandomizedSearchCV, é necessário fornecer os dados como arrays
    # Uma alternativa é carregar os dados completamente na memória se possível
    # Caso contrário, considere usar outras abordagens como Bayesian Optimization

    # Carregar todos os dados em memória (se possível)
    with h5py.File(HDF5_FILE, 'r') as h5f:
        X = h5f['data'][:]
        y = h5f['labels'][:]

    # Executar a busca de hiperparâmetros
    print("Iniciando a busca de hiperparâmetros...")
    search_result = search_method.fit(X, y)
    print("Busca de hiperparâmetros concluída.")

    # Exibir os melhores resultados
    print(f"Melhor F1 ponderado: {search_result.best_score_ * 100:.2f}%")
    print("Melhores parâmetros: ", search_result.best_params_)

    # Salvar os resultados em um arquivo de texto
    save_results_to_txt(search_result)
    print("Resultados salvos em 'results/tuning_results.txt'.")

if __name__ == "__main__":
    main()
