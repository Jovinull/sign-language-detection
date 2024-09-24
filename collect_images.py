import os
import cv2
from image_capture_utils.video_capture import start_video_capture
from image_capture_utils.image_storage import get_existing_image_count
from image_capture_utils.hand_detection import process_frame

DATA_DIR = './data'
number_of_classes = 21
dataset_size = 50
IMG_SIZE = 512

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Abre a captura de vídeo (webcam)
cap = cv2.VideoCapture(0)

# Escolher se deseja capturar imagens para uma classe específica ou para todas
option = input("Deseja capturar para uma classe específica (digite '1') ou para todas as classes (digite '2')? ")

if option == '1':
    start_video_capture(cap, DATA_DIR, number_of_classes, dataset_size, IMG_SIZE, specific_class=True)
elif option == '2':
    start_video_capture(cap, DATA_DIR, number_of_classes, dataset_size, IMG_SIZE, specific_class=False)
else:
    print("Opção inválida! Por favor, digite '1' para capturar uma classe ou '2' para capturar todas as classes.")

cap.release()
cv2.destroyAllWindows()
