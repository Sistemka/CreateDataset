import json
import os
from os.path import isfile
from time import time

from PIL import Image
import tensorflow as tf
import numpy as np


root_dir = 'files'


def is_image(file_name):
    image_formats = {'png', 'jpeg', 'jpg'}
    try:
        if file_name.split('.')[-1].lower() in image_formats:
            return True
        return False
    except Exception:
        return False


def make_square(img, min_size=224, fill_color=(255, 255, 255, 0)) -> Image:
    """
    Делает картинку квадратной с сохранением пропорций
    """
    output_size = min_size, min_size
    x, y = img.size
    max_size = max(x, y)
    coeff = min_size / max_size
    x, y = round(x * coeff), round(y * coeff)
    img = img.resize((x, y), 1)
    size = max(min_size, x, y)
    new_im = Image.new('RGB', (size, size), fill_color)
    new_im.paste(img, (int((size - x) / 2), int((size - y) / 2)))
    new_im.thumbnail(output_size, Image.ANTIALIAS)
    return new_im


def img2vec_that_saves_proportions(image_path):
    return tf.keras.preprocessing.image.img_to_array(img=make_square(Image.open(image_path)))


def make_vectors_from_all(root_path=root_dir) -> list:
    dirs = os.listdir(root_path)
    final_list = []
    for folder in dirs:
        for file in os.listdir(os.path.join(root_path, folder)):
            full_file_path = os.path.join(root_path, folder, file)
            if isfile(full_file_path):
                final_list.append({
                    "type": folder.split('/')[-1],
                    "vector": img2vec_that_saves_proportions(full_file_path)
                })
    return final_list


def make_vectors_from_all_and_return_json(root_path=root_dir) -> list:
    dirs = os.listdir(root_path)
    final_list = []
    for folder in dirs:
        for file in os.listdir(os.path.join(root_path, folder)):
            full_file_path = os.path.join(root_path, folder, file)
            if is_image(full_file_path):
                final_list.append({
                    "type": folder.split('/')[-1],
                    "vector": str(img2vec_that_saves_proportions(full_file_path))
                })
    return final_list


def make_vectors_from_all_and_return_dict(root_path=root_dir) -> dict:
    dirs = os.listdir(root_path)
    final_list = {}
    for folder in dirs:
        for file in os.listdir(os.path.join(root_path, folder)):
            full_file_path = os.path.join(root_path, folder, file)
            if is_image(full_file_path):
                if final_list.get(folder.split('/')[-1]):
                    final_list[folder.split(
                        '/')[-1]].append(img2vec_that_saves_proportions(full_file_path))
                else:
                    final_list[folder.split(
                        '/')[-1]] = [img2vec_that_saves_proportions(full_file_path)]
    return final_list


def auto_set_comparison(root_path=root_dir):
    """
    Автоматически проставляет похожесть.
    Объект похож только сам на себя (1)
    На остальное не похож совсем (0)
    """
    dirs = os.listdir(root_path)
    for folder in dirs:
        comparing_dict = {}
        for comparing_dir in dirs:
            if folder == comparing_dir:
                comparing_dict[comparing_dir] = 1
            else:
                comparing_dict[comparing_dir] = 0
        with open(os.path.join(root_path, folder, "compare.txt"), 'w') as file:
            file.write(json.dumps(comparing_dict))


def create_comparing_vectors(root_path=root_dir):
    """
    Составляет 3 вектора:
    1) и 2) все возможные сочетания лейблов
    3) веса составленные на основе compare.txt, который находится в каждой папке
    """
    x_left = []
    x_right = []
    weigts_vector = []
    all_vectors = make_vectors_from_all_and_return_dict(
        root_path)  # обсудить что надо принимать
    all_labels = all_vectors.keys()
    for label in all_labels:
        x_left.extend(all_vectors.get(label) * len(all_labels))
        with open(os.path.join(root_path, label, "compare.txt"), "r") as file:
            comparison = json.load(file)
            comparing_labels = comparison.keys()
            for comparing_label in comparing_labels:
                x_right.extend(all_vectors.get(comparing_label))
                weigts_vector.extend([comparison.get(comparing_label)] * 10)
    return np.array(x_left), np.array(x_right), np.array(weigts_vector)


def add_label(label_name, comparing_dict):
    """
    Добавляет новый лейбл и прописывает указанные веса в новый и предыдущие лейблы
    """
    if type(comparing_dict) != dict:
        raise ValueError("""
            Please compare your label with existing.
            Existing labels are: %s 
            Input has to be like:
            {'#label#' : "#weight#"}
            F.ex : {"boots" : 0}
            """ % os.listdir(root_dir))
    if len(comparing_dict) != len(os.listdir(root_dir)):
        raise ValueError("""
            Please compare all existing labels with yours
            Number of existing labels is %s
            """ % len(os.listdir(root_dir)))
    current_labels = set(os.listdir(root_dir))
    for map_label in comparing_dict.keys():
        if map_label not in current_labels:
            raise ValueError("""
                Label that you are trying to compare with does not exist
                Error label: %s
                Existing labels: %s
                """ % (map_label, os.listdir(root_dir)))
    comparing_dict[label_name] = 1
    for prev_label in os.listdir(root_dir):
        with open(os.path.join(root_dir, prev_label, 'compare.txt'), 'r+') as file:
            weights_map = json.load(file)
            file.seek(0)
            file.truncate()
            weights_map[label_name] = comparing_dict.get(prev_label)
            file.write(json.dumps(weights_map))
            file.close()
    os.mkdir(os.path.join(root_dir, label_name))
    with open(os.path.join(root_dir, label_name, "compare.txt"), "w") as file:
        file.write(json.dumps(comparing_dict))
        file.close()


def add_picture(img, label):
    current_labels = set(os.listdir(root_dir))
    if label not in current_labels:
        raise ValueError("""
            Label that you are trying to save to does not exist
            Error label: %s
            Existing labels: %s
            """ % (label, os.listdir(root_dir)))
    img.save(os.path.join(root_dir, label, str(hash(time())) + ".png"))
