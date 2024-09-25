import cv2

def resize_image(image, target_size):
    # Obter as dimensões da imagem original
    height, width = image.shape[:2]
    
    # Calcular a proporção
    aspect_ratio = width / height

    # Definir as novas dimensões mantendo a proporção
    if aspect_ratio > 1:  # Largura maior que altura
        new_width = target_size
        new_height = int(target_size / aspect_ratio)
    else:  # Altura maior que largura
        new_height = target_size
        new_width = int(target_size * aspect_ratio)

    # Redimensionar a imagem
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    return resized_image
