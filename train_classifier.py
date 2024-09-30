import tensorflow as tf
from sklearn.model_selection import KFold
from train_utils.data_loader import load_data_hdf5, get_train_test_split
from train_utils.model_builder import create_model
import config
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import os

# Função para realizar o treinamento com validação cruzada
def cross_validate_and_train(h5f, labels, n_splits, BATCH_SIZE, epochs):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_no = 1
    metrics_summary = []

    for train_index, test_index in kf.split(labels):
        print(f'\n--- Iniciando a fold {fold_no} ---')

        # Obtém os dados de treino e teste para esta fold
        x_train, y_train, x_test, y_test = get_train_test_split(h5f, labels, train_index, test_index)

        # Cria o modelo
        model = create_model()

        print(f'Treinando a fold {fold_no}...')

        # Define os datasets utilizando tf.data para otimização
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        train_dataset = train_dataset.shuffle(buffer_size=1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

        test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

        # Treina o modelo
        history = model.fit(
            train_dataset,
            epochs=epochs,
            validation_data=test_dataset,
            verbose=1
        )

        # Faz previsões no conjunto de teste
        y_pred = model.predict(test_dataset)
        y_pred_classes = np.argmax(y_pred, axis=1)

        # Calcula precisão, recall e F1-score
        precision = precision_score(y_test, y_pred_classes, average='weighted')
        recall = recall_score(y_test, y_pred_classes, average='weighted')
        f1 = f1_score(y_test, y_pred_classes, average='weighted')

        # Avalia a acurácia no conjunto de teste
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
    os.makedirs('results', exist_ok=True)
    model.save('results/hand_gesture_cnn_kfold.h5')

    return metrics_summary

def main():
    # Carregar os dados a partir do arquivo HDF5
    print("Carregando dados do arquivo HDF5...")
    h5f, labels = load_data_hdf5()
    print(f"Dados carregados: {labels.shape[0]} exemplos.")

    # Verificar se os dados precisam ser embaralhados
    # Como o KFold já está com shuffle=True, não é necessário embaralhar novamente

    # Executar a validação cruzada e treinamento
    metrics_summary = cross_validate_and_train(
        h5f,
        labels,
        config.N_SPLITS,
        config.BATCH_SIZE,
        config.epochs
    )

    # Fechar o arquivo HDF5
    h5f.close()

    # Exibir um resumo das métricas
    print("\n--- Resumo das Métricas por Fold ---")
    for metric in metrics_summary:
        print(f"Fold {metric['fold']}: Acurácia={metric['accuracy']*100:.2f}%, "
              f"Precisão={metric['precision']:.4f}, Recall={metric['recall']:.4f}, "
              f"F1-Score={metric['f1_score']:.4f}")

    # Salvar o resumo das métricas em um arquivo
    os.makedirs('results', exist_ok=True)
    with open('results/metrics_summary.txt', 'w') as f:
        for metric in metrics_summary:
            f.write(f"Fold {metric['fold']}: Acurácia={metric['accuracy']*100:.2f}%, "
                    f"Precisão={metric['precision']:.4f}, Recall={metric['recall']:.4f}, "
                    f"F1-Score={metric['f1_score']:.4f}\n")

    print("\nResumo das métricas salvo em 'results/metrics_summary.txt'.")

if __name__ == "__main__":
    main()
