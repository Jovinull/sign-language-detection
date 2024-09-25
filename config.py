# Parâmetros de dados
DATA_DIR = './data'
IMG_SIZE = 512  # Tamanho das imagens

# Parâmetros de treino
N_SPLITS = 10  # Número de divisões para a validação cruzada (FOLDS)
BATCH_SIZE = 35  # # Quantidade de amostras processadas antes de atualizar os pesos
epochs = 10  # Número de épocas

# BATCH_SIZE Possíveis candidatos
# 35, 50, 70, 84, 100, 140 e 175

# N_SPLITS: número de divisões para a validação cruzada (FOLDS). O modelo é treinado N_SPLITS vezes, 
#           cada vez com um subconjunto diferente dos dados de treino, para garantir uma avaliação mais robusta.
# BATCH_SIZE: quantidade de amostras processadas antes de ajustar os pesos do modelo. 
#             O modelo calcula o erro para o batch e ajusta os pesos com base nisso.
# epochs: número de vezes que o modelo passa por todo o conjunto de dados. Em cada época, o modelo ajusta os pesos 
#         várias vezes (dependendo do batch size) até percorrer todo o dataset. Mais épocas permitem que o modelo aprenda mais.

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
    "conv_layers": [2, 3, 4],
    "filters": [16, 32, 64],
    "kernel_size": [3, 5],
    "dense_units": [128, 256],
    "dropout_rate": [0.3, 0.5],
    "activation": ['relu', 'tanh'],
    "optimizer": ['adam', 'sgd'],
    "l2_reg": [0.001, 0.01],
    "epochs": [50],
    "batch_size": [35, 84, 140]
}

# Parâmetros de busca (Grid ou Random)
search_type = 'random'  # Pode ser 'grid' ou 'random'
n_iter = 2  # Usado apenas para RandomizedSearchCV
