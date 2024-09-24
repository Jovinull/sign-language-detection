import os
import cv2
from image_capture_utils.image_storage import get_existing_image_count
from image_capture_utils.hand_detection import process_frame

def check_key_press(key='q'):
    return cv2.waitKey(1) & 0xFF == ord(key)

def start_video_capture(cap, DATA_DIR, number_of_classes, dataset_size, IMG_SIZE, specific_class=None):
    classes_to_capture = [specific_class] if specific_class is not None else range(number_of_classes)

    for class_number in classes_to_capture:
        class_dir = os.path.join(DATA_DIR, str(class_number))
        os.makedirs(class_dir, exist_ok=True)
        print(f'Pressione "Q" para iniciar a captura para a classe {class_number}.')
        capture_images_for_class(cap, class_dir, class_number, dataset_size, IMG_SIZE)

def capture_images_for_class(cap, class_dir, class_number, dataset_size, IMG_SIZE):
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, 'Press "Q" to start', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('frame', frame)
        if check_key_press('q'):
            break

    existing_images = get_existing_image_count(class_dir)
    counter = 0

    while counter < dataset_size:
        ret, frame = cap.read()
        hand_resized, frame_with_box = process_frame(frame, IMG_SIZE)

        if hand_resized is not None and hand_resized.shape[0] > 50 and hand_resized.shape[1] > 50:
            cv2.imwrite(os.path.join(class_dir, '{}.jpg'.format(existing_images + counter)), hand_resized)
            counter += 1
            print(f'Capturando imagem {counter}/{dataset_size} para a classe {class_number}')
            cv2.putText(frame_with_box, f'Class {class_number}: {counter}/{dataset_size}', (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.imshow('frame', frame_with_box)
        else:
            cv2.imshow('frame', frame_with_box)

        if check_key_press('q'):
            break

    print(f'Concluída a captura para a classe {class_number}')
