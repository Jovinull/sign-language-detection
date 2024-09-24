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

# Escolher se deseja capturar imagens para uma classe específica ou para todas
option = input("Deseja capturar para uma classe específica (digite '1') ou para todas as classes (digite '2')? ")

if option == '1':
    # Definir a classe para a qual deseja capturar imagens
    class_to_capture = int(input(f"Informe o número da classe para capturar imagens (0 a {number_of_classes - 1}): "))

    # Verifica se o número da classe está dentro do intervalo válido
    if 0 <= class_to_capture < number_of_classes:
        class_dir = os.path.join(DATA_DIR, str(class_to_capture))
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)

        print(f'Pressione "Q" para iniciar a captura para a classe {class_to_capture}.')

        # Exibe instrução antes de começar a captura
        while True:
            ret, frame = cap.read()
            cv2.putText(frame, 'Press "Q" to start', (50, 50),
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
            cv2.putText(frame, 'Collecting for class {}: {}/{}'.format(class_to_capture, counter, dataset_size), (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.imshow('frame', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):  # Permite parar a captura com 'Q'
                break

        print(f'Concluída a captura para a classe {class_to_capture}')

    else:
        print(f"Classe inválida! Por favor, insira um número entre 0 e {number_of_classes - 1}.")

elif option == '2':
    # Loop para capturar imagens para todas as classes
    for j in range(number_of_classes):
        class_dir = os.path.join(DATA_DIR, str(j))
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)

        print(f'Pressione "Q" para iniciar a captura para a classe {j}.')

        # Exibe instrução antes de começar a captura
        while True:
            ret, frame = cap.read()
            cv2.putText(frame, 'Press "Q" to start', (50, 50),
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

        print(f'Concluída a captura para a classe {j}')

else:
    print("Opção inválida! Por favor, digite '1' para capturar uma classe ou '2' para capturar todas as classes.")

cap.release()
cv2.destroyAllWindows()
