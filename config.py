# Parâmetros de dados
DATA_DIR = './data'
IMG_SIZE = 128

# Parâmetros do modelo
conv_layers = 3
filters = 32
kernel_size = 3
dense_units = 512
dropout_rate = 0.5
activation = 'relu'
optimizer = 'adam'
epochs = 10
batch_size = 32

# Parâmetros de busca
param_grid = {
    "conv_layers": [2, 3],
    "filters": [32, 64, 128],
    "kernel_size": [3, 5],
    "dense_units": [128, 256],
    "dropout_rate": [0.3, 0.5],
    "activation": ['relu', 'tanh'],
    "optimizer": ['adam', 'sgd'],
    "epochs": [50],
    "batch_size": [32, 64]
}

# Parâmetros de busca (Grid ou Random)
search_type = 'random'  # Pode ser 'grid' ou 'random'
n_iter = 2  # Usado apenas para RandomizedSearchCV
