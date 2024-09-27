import os
import numpy as np
import cv2
from keras import layers, models, regularizers
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.utils import shuffle
from scikeras.wrappers import KerasClassifier
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
import config  # Importar os parâmetros do arquivo config.py
from utils.resize_image_by_aspect import resize_image  # Função para redimensionar as imagens

# Carregando e processando os dados
DATA_DIR = config.DATA_DIR

def load_data(img_size):
    data = []
    labels = []
    
    for class_dir in os.listdir(DATA_DIR):
        class_path = os.path.join(DATA_DIR, class_dir)
        label = int(class_dir)
        
        for img_path in os.listdir(class_path):
            img = cv2.imread(os.path.join(class_path, img_path))
            img = resize_image(img, img_size)  # Redimensiona com o tamanho dinâmico
            data.append(img)
            labels.append(label)
    
    data = np.array(data, dtype="float32") / 255.0
    labels = np.array(labels)
    
    return data, labels

def create_model(conv_layers, filters, kernel_size, dense_units, dropout_rate, activation, optimizer, l2_reg, img_size):
    if isinstance(filters, int):
        filters = [filters] * conv_layers  # Cria uma lista com o mesmo valor para cada camada

    model = models.Sequential()

    # Camadas convolucionais dinamicamente com batch normalization
    for i in range(conv_layers):
        filters_layer = filters[i]  # Ajuste dinâmico de filtros por camada
        if i == 0:
            model.add(layers.Conv2D(filters_layer, (kernel_size, kernel_size), activation=activation,
                                    input_shape=(img_size, img_size, 3),
                                    kernel_regularizer=regularizers.l2(l2_reg)))
        else:
            model.add(layers.Conv2D(filters_layer, (kernel_size, kernel_size), activation=activation,
                                    kernel_regularizer=regularizers.l2(l2_reg)))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2, 2)))

    # Global pooling ao invés de Flatten
    model.add(layers.GlobalAveragePooling2D())

    # Camada densa e saída
    model.add(layers.Dense(dense_units, activation=activation,
                           kernel_regularizer=regularizers.l2(l2_reg)))
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(20, activation='softmax'))

    # Compilando o modelo
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Modelo KerasClassifier com hiperparâmetros
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
    batch_size=config.BATCH_SIZE,
    img_size=config.IMG_SIZE,  # Passar o img_size dinamicamente
    verbose=1
)

# Parâmetros de tuning
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
    'batch_size': config.param_grid['batch_size'],
    'model__img_size': config.param_grid['IMG_SIZE']
}

# Definindo uma função para escolher o método de busca e aplicar as métricas adicionais
def choose_search_method(search_type=config.search_type, n_iter=config.n_iter):
    # Definindo os scorers (métricas) personalizados
    scoring = {
        'accuracy': 'accuracy',
        'precision_weighted': make_scorer(precision_score, average='weighted'),
        'recall_weighted': make_scorer(recall_score, average='weighted'),
        'f1_weighted': make_scorer(f1_score, average='weighted')
    }
    
    if search_type == 'grid':
        print("Usando GridSearchCV...")
        search = GridSearchCV(estimator=model, param_grid=param_grid, cv=config.N_SPLITS, 
                              verbose=2, scoring=scoring, refit='f1_weighted')  # Refit com F1 ponderado
    elif search_type == 'random':
        print(f"Usando RandomizedSearchCV com {n_iter} iterações...")
        search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=n_iter, 
                                    cv=config.N_SPLITS, verbose=2, random_state=42, scoring=scoring, 
                                    refit='f1_weighted')  # Refit com F1 ponderado
    else:
        raise ValueError("search_type deve ser 'grid' ou 'random'.")
    
    return search

# Carregar os dados com o tamanho inicial
img_size = config.IMG_SIZE  # Tamanho inicial, mas será ajustado no tuning
data, labels = load_data(img_size)
data, labels = shuffle(data, labels, random_state=42)

# Executar a busca de hiperparâmetros
search_method = choose_search_method()
search_result = search_method.fit(data, labels)

print(f"Melhor F1 ponderado: {search_result.best_score_ * 100:.2f}%")
print("Melhores parâmetros: ", search_result.best_params_)

# Função para salvar os resultados em um arquivo
def save_results_to_txt(search_result, filename="results/tuning_results.txt"):
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

save_results_to_txt(search_result)
