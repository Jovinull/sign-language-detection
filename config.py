# Parâmetros de dados
DATA_DIR = './data'
IMG_SIZE = 512  # Tamanho das imagens

# Parâmetros de treino
N_SPLITS = 10  # Número de divisões para a validação cruzada
BATCH_SIZE = 35  # Tamanho do lote
epochs = 10  # Número de épocas

# BATCH_SIZE Possíveis candidatos
# 35, 50, 70, 84, 100, 140

# Parâmetros do modelo
conv_layers = 3
filters = [16, 32, 64]  # Filtros para cada camada convolucional
kernel_size = 3  # Tamanho do kernel (filtro)
dense_units = 128  # Número de unidades na camada densa
dropout_rate = 0.3  # Taxa de dropout
activation = 'relu'  # Função de ativação
optimizer = 'adam'  # Otimizador
l2_reg = 0.001  # Regularização L2

# Parâmetros de tuning (busca de hiperparâmetros)
param_grid = {
    "conv_layers": [2, 3],
    "filters": [32, 64, 128],
    "kernel_size": [3, 5],
    "dense_units": [128, 256],
    "dropout_rate": [0.3, 0.5],
    "activation": ['relu', 'tanh'],
    "optimizer": ['adam', 'sgd'],
    "l2_reg": [0.001, 0.01],
    "epochs": [50],
    "batch_size": [32, 64]
}

# Parâmetros de busca (Grid ou Random)
search_type = 'random'  # Pode ser 'grid' ou 'random'
n_iter = 2  # Usado apenas para RandomizedSearchCV
