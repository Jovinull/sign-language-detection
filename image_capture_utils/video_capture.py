import os
import cv2
from image_capture_utils.image_storage import get_existing_image_count
from image_capture_utils.hand_detection import process_frame

def start_video_capture(cap, DATA_DIR, number_of_classes, dataset_size, IMG_SIZE, specific_class=False):
    if specific_class:
        class_to_capture = int(input(f"Informe o número da classe para capturar imagens (0 a {number_of_classes - 1}): "))
        if 0 <= class_to_capture < number_of_classes:
            class_dir = os.path.join(DATA_DIR, str(class_to_capture))
            if not os.path.exists(class_dir):
                os.makedirs(class_dir)

            print(f'Pressione "Q" para iniciar a captura para a classe {class_to_capture}.')
            capture_images_for_class(cap, class_dir, class_to_capture, dataset_size, IMG_SIZE)
        else:
            print(f"Classe inválida! Insira um número entre 0 e {number_of_classes - 1}.")
    else:
        for j in range(number_of_classes):
            class_dir = os.path.join(DATA_DIR, str(j))
            if not os.path.exists(class_dir):
                os.makedirs(class_dir)

            print(f'Pressione "Q" para iniciar a captura para a classe {j}.')
            capture_images_for_class(cap, class_dir, j, dataset_size, IMG_SIZE)

def capture_images_for_class(cap, class_dir, class_number, dataset_size, IMG_SIZE):
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, 'Press "Q" to start', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    existing_images = get_existing_image_count(class_dir)
    counter = 0

    while counter < dataset_size:
        ret, frame = cap.read()
        hand_resized, frame_with_box = process_frame(frame, IMG_SIZE)

        if hand_resized is not None:
            cv2.imwrite(os.path.join(class_dir, '{}.jpg'.format(existing_images + counter)), hand_resized)
            counter += 1

            cv2.putText(frame_with_box, f'Class {class_number}: {counter}/{dataset_size}', (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.imshow('frame', frame_with_box)
        else:
            cv2.imshow('frame', frame_with_box)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print(f'Concluída a captura para a classe {class_number}')
