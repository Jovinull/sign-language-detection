import os
import cv2

# Diretório onde os dados serão armazenados
DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 21  # Ajustar de acordo com a quantidade de classes
dataset_size = 100  # Ajustar para a quantidade de imagens por classe

# Função para contar quantos arquivos já existem na pasta da classe
def get_existing_image_count(class_dir):
    return len([name for name in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, name))])

# Abre a captura de vídeo (webcam)
cap = cv2.VideoCapture(0)

for j in range(number_of_classes):
    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print(f'Press "Q" when ready.')

    # Exibe instrução antes de começar a captura
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, 'Press "Q" to start'.format(j), (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Conta quantos arquivos já existem para evitar sobrescrever
    existing_images = get_existing_image_count(class_dir)
    counter = 0

    while counter < dataset_size:
        ret, frame = cap.read()

        # Salva a imagem no tamanho original da webcam
        cv2.imwrite(os.path.join(class_dir, '{}.jpg'.format(existing_images + counter)), frame)
        counter += 1

        # Mostra o progresso
        cv2.putText(frame, 'Collecting for class {}: {}/{}'.format(j, counter, dataset_size), (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Permite parar a captura com 'Q'
            break

    print(f'Done collecting for class {j}')

cap.release()
cv2.destroyAllWindows()
