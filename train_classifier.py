import tensorflow as tf
from sklearn.model_selection import KFold
from train_utils.data_loader import load_data
from train_utils.model_builder import create_model
import config
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

# Função para realizar o treinamento com validação cruzada
def cross_validate_and_train(data, labels, n_splits, BATCH_SIZE, epochs):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_no = 1
    metrics_summary = []

    for train_index, test_index in kf.split(data):
        x_train, x_test = data[train_index], data[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        model = create_model()

        print(f'Treinando a fold {fold_no}...')

        # Definir os datasets
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

        # Treinar o modelo
        history = model.fit(train_dataset, epochs=epochs, validation_data=test_dataset)

        # Fazer previsões no conjunto de teste
        y_pred = model.predict(test_dataset)
        y_pred_classes = np.argmax(y_pred, axis=1)

        # Calcular precisão, recall e F1-score
        precision = precision_score(y_test, y_pred_classes, average='weighted')
        recall = recall_score(y_test, y_pred_classes, average='weighted')
        f1 = f1_score(y_test, y_pred_classes, average='weighted')

        # Avaliar a acurácia no conjunto de teste
        accuracy = np.mean(y_test == y_pred_classes)

        print(f'Acurácia da fold {fold_no}: {accuracy * 100:.2f}%')
        print(f'Precisão da fold {fold_no}: {precision:.4f}')
        print(f'Recall da fold {fold_no}: {recall:.4f}')
        print(f'F1-Score da fold {fold_no}: {f1:.4f}')

        metrics_summary.append({
            'fold': fold_no,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        })

        fold_no += 1

    # Salvando o último modelo treinado
    model.save('results/hand_gesture_cnn_kfold.h5')

    return metrics_summary


# Carregar dados e executar o treinamento diretamente ao rodar este arquivo
data, labels = load_data()  # Agora retornando data e labels normalmente
cross_validate_and_train(data, labels, config.N_SPLITS, config.BATCH_SIZE, config.epochs)
