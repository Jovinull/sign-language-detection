import os

def get_existing_image_count(class_dir):
    return len([name for name in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, name))])
