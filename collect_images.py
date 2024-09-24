import os
import cv2
from image_capture_utils.video_capture import start_video_capture

DATA_DIR = './data'
number_of_classes = 21
dataset_size = 50
IMG_SIZE = 512

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Abre a captura de vídeo (webcam)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erro ao acessar a webcam.")
    exit()

# Escolher se deseja capturar imagens para uma classe específica ou para todas
option = input("Deseja capturar para uma classe específica (digite '1') ou para todas as classes (digite '2')? ")

if option == '1':
    class_num = int(input(f"Informe o número da classe para capturar imagens (0 a {number_of_classes - 1}): "))
    if 0 <= class_num < number_of_classes:
        start_video_capture(cap, DATA_DIR, number_of_classes, dataset_size, IMG_SIZE, specific_class=class_num)
    else:
        print(f"Classe inválida! Insira um número entre 0 e {number_of_classes - 1}.")
elif option == '2':
    start_video_capture(cap, DATA_DIR, number_of_classes, dataset_size, IMG_SIZE)
else:
    print("Opção inválida! Por favor, digite '1' ou '2'.")

cap.release()
cv2.destroyAllWindows()
