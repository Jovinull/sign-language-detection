import os
import numpy as np
import cv2
from keras import layers, models
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.utils import shuffle
from scikeras.wrappers import KerasClassifier
import config  # Importar os parâmetros do arquivo config.py
from keras import regularizers

# Carregando e processando os dados
DATA_DIR = config.DATA_DIR
IMG_SIZE = config.IMG_SIZE

def load_data():
    data = []
    labels = []
    
    for class_dir in os.listdir(DATA_DIR):
        class_path = os.path.join(DATA_DIR, class_dir)
        label = int(class_dir)
        
        for img_path in os.listdir(class_path):
            img = cv2.imread(os.path.join(class_path, img_path))
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            data.append(img)
            labels.append(label)
    
    data = np.array(data, dtype="float32") / 255.0
    labels = np.array(labels)
    
    return data, labels

data, labels = load_data()
data, labels = shuffle(data, labels, random_state=42)

def create_model(conv_layers=config.conv_layers, filters=config.filters, kernel_size=config.kernel_size,
                 dense_units=config.dense_units, dropout_rate=config.dropout_rate,
                 activation=config.activation, optimizer=config.optimizer,
                 l2_reg=config.l2_reg):  # Novo parâmetro l2_reg
    
    model = models.Sequential()
    model.add(layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)))
    
    for i in range(conv_layers):
        model.add(layers.Conv2D(filters, (kernel_size, kernel_size), activation=activation,
                                kernel_regularizer=regularizers.l2(l2_reg)))  # Aplicando L2
        model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(dense_units, activation=activation,
                           kernel_regularizer=regularizers.l2(l2_reg)))  # Aplicando L2
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(21, activation='softmax'))

    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

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
    batch_size=config.batch_size,
    verbose=1
)

# Parâmetros de busca
param_grid = config.param_grid

def choose_search_method(search_type=config.search_type, n_iter=config.n_iter):
    if search_type == 'grid':
        print("Usando GridSearchCV...")
        search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, verbose=2)
    elif search_type == 'random':
        print(f"Usando RandomizedSearchCV com {n_iter} iterações...")
        search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=n_iter, cv=5, verbose=2, random_state=42)
    else:
        raise ValueError("search_type deve ser 'grid' ou 'random'.")
    
    return search

search_method = choose_search_method()

search_result = search_method.fit(data, labels)

print(f"Melhor acurácia: {search_result.best_score_ * 100:.2f}%")
print("Melhores parâmetros: ", search_result.best_params_)

def save_results_to_txt(search_result, filename="results/tuning_results.txt"):
    # Garantir que o diretório exista
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, "w") as file:
        file.write("Melhores parâmetros:\n")
        file.write(str(search_result.best_params_) + "\n\n")
        
        file.write(f"Melhor acurácia: {search_result.best_score_ * 100:.2f}%\n\n")
        
        for i, candidate in enumerate(search_result.cv_results_['params']):
            file.write(f"Configuração {i+1}: {candidate}\n")
            file.write(f"Acurácia média: {search_result.cv_results_['mean_test_score'][i]:.4f}\n")
            file.write(f"Desvio padrão da acurácia: {search_result.cv_results_['std_test_score'][i]:.4f}\n\n")

save_results_to_txt(search_result)
