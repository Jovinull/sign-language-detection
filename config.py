# Configurações Gerais
DATA_DIR = './data'  # Atualize com o caminho real dos seus dados
IMG_SIZE = 224  # Exemplo: 224x224 pixels
N_SPLITS = 5
BATCH_SIZE = 125
epochs = 20

#################################################################################################################################
# N_SPLITS: número de divisões para a validação cruzada (FOLDS). O modelo é treinado N_SPLITS vezes,                            #
#           cada vez com um subconjunto diferente dos dados de treino, para garantir uma avaliação mais robusta.                #
# BATCH_SIZE: quantidade de amostras processadas antes de ajustar os pesos do modelo.                                           #
#             O modelo calcula o erro para o batch e ajusta os pesos com base nisso.                                            #
#             # Possíveis Candidatos: 40, 125, 250, 500                                               #
# epochs: número de vezes que o modelo passa por todo o conjunto de dados. Em cada época, o modelo ajusta os pesos              #
#         várias vezes (dependendo do batch size) até percorrer todo o dataset. Mais épocas permitem que o modelo aprenda mais. #
#################################################################################################################################

# Redimensionamento
INTERPOLATION_METHOD = 'INTER_AREA'  # Opções: 'INTER_AREA', 'INTER_LINEAR', 'INTER_CUBIC'

# Modelo
conv_layers = 3
filters = [32, 64, 128]
kernel_size = 3
activation = 'relu'
l2_reg = 0.001
dense_units = 256
dropout_rate = 0.5
optimizer = 'adam'

# Tuning
search_type = 'random'  # 'grid' ou 'random'
n_iter = 20  # Número de iterações para RandomizedSearchCV
param_grid = {
    'conv_layers': [2, 3, 4],
    'filters': [[32, 64], [64, 128], [128, 256]],
    'kernel_size': [3, 5],
    'dense_units': [128, 256, 512],
    'dropout_rate': [0.3, 0.5, 0.7],
    'activation': ['relu', 'tanh'],
    'optimizer': ['adam', 'sgd'],
    'l2_reg': [0.0001, 0.001, 0.01],
    'epochs': [10, 20],
    'batch_size': [32, 64]
}
