import os

import cv2
import numpy as np
from PIL import Image
from rembg import remove


""" Aqui se agregan todas las clases o funciones para realizar el pre-procesamiento de los datos """
class ImageTransformation:
    def __init__(self, input_dir, categories):
        self.input_dir = input_dir
        self.categories = categories

    def remove_background(self, output_dir):
        for category_idx, category in enumerate(self.categories):
            for file in os.listdir(os.path.join(self.input_dir, category)):
                output_path = os.path.join(output_dir,
                                           category,
                                           file[:-4] + ".png")
                img_path = os.path.join(self.input_dir, category, file)
                input_image = Image.open(img_path)
                rgb_im = input_image.convert("RGB")
                output_image = remove(rgb_im)
                output_image.save(output_path)

    def image_normalize(self):
        data_sin_plagas = []
        data_plagas = []
        labels_sin_plagas = []
        labels_plagas = []

        for category_idx, category in enumerate(self.categories):
            for file in os.listdir(os.path.join(self.input_dir, category)):
                img_path = os.path.join(self.input_dir, category, file)

                img = cv2.imread(img_path)
                image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                scaled_image = cv2.resize(src=image_rgb, dsize=(640, 640))
                if category == "sin_plaga":
                    data_sin_plagas.append(scaled_image.flatten())
                    labels_sin_plagas.append(category_idx)
                else:
                    data_plagas.append(scaled_image.flatten())
                    labels_plagas.append(category_idx)

        data_sin_plagas = np.asarray(data_sin_plagas)
        labels_sin_plagas = np.asarray(labels_sin_plagas)
        data_plagas = np.asarray(data_plagas)
        labels_plagas = np.asarray(labels_plagas)

        return data_sin_plagas, labels_sin_plagas, data_plagas, labels_plagas