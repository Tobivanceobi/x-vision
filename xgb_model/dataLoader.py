import io
import os
import zipfile

import joblib
import numpy as np
import pandas as pd
from PIL import Image

from utils import load_pickle, save_pickle


def preprocess_image(image):
    img = image.convert('L').resize((100, 100))
    img_array = np.array(img).flatten()
    return img_array


class DataLoader:
    BASE_PATH = r'/run/media/tobias/Archiv/CheXpert_Dataset/chexpertchestxrays-u20210408/'
    CACHE_PATH = r'cache/'
    XRAY_VIEW = 'Frontal'
    XRAY_DIREC = 'AP'

    def __init__(self, use_cache=True):
        self.cont_df = self.load_data_contents()
        self.img_data = []
        self.img_labels = []
        self.__image_files = []
        self.__use_cache = use_cache

        self.__cache_list = []
        if self.__use_cache and os.path.exists(self.CACHE_PATH + 'cache_list.pickle'):
            self.__cache_list = load_pickle(self.CACHE_PATH + 'cache_list.pickle')

    def load_data_contents(self):
        print("DataLoader: Loading image contents.")
        df = pd.read_csv(self.BASE_PATH + 'train_cheXbert.csv')
        return df

    def get_image_paths(self, config: dict):
        print("DataLoader: Finding selected file paths")
        for lab in config.keys():
            if not (lab in self.cont_df.columns.values):
                print(f"ERROR: Requested data label - {lab} - not found in dataset")
                continue
            lab_data = self.cont_df[
                (self.cont_df[lab] == 1) &
                (self.cont_df['Frontal/Lateral'] == self.XRAY_VIEW) &
                (self.cont_df['AP/PA'] == self.XRAY_DIREC)
            ].head(config[lab])
            file_paths = lab_data['Path'].values.tolist()
            for fp in file_paths:
                self.__image_files.append(fp)
                self.img_labels.append(lab)
        self.img_labels = np.array(self.img_labels)
        return self.__image_files

    def load_images_from_files(self, preprocess=True):
        print("DataLoader: Loading image data from file paths.")
        if not self.__image_files:
            print("No images to load.")
            return None

        def load_image(file):
            print(f"DataLoader:     Loading {file + 1} of {len(self.__image_files)} files.")
            fname = self.CACHE_PATH + self.__image_files[file].replace("/", "_").replace(".jpg", "")+".pickle"
            if os.path.exists(fname):
                return load_pickle(fname)
            img = self.load_jpeg_from_zip(self.__image_files[file])
            if img:
                if preprocess:
                    img = preprocess_image(img)
                    save_pickle(img, fname)
                    self.__cache_list.append(self.__image_files[file])
                    return img
                return img
            return None

        self.img_data = joblib.Parallel(n_jobs=15)(
            joblib.delayed(load_image)(file) for file in range(len(self.__image_files)))
        self.img_data = np.array([img for img in self.img_data if img is not None])
        save_pickle(self.__cache_list, self.CACHE_PATH + 'cache_list.pickle')
        return self.img_data, self.img_labels

    def load_jpeg_from_zip(self, image_filename):
        zip_file_path = self.BASE_PATH + 'CheXpert-v1.0.zip'

        try:
            with zipfile.ZipFile(zip_file_path, 'r') as zip_file:
                with zip_file.open(image_filename) as image_file:
                    image_data = image_file.read()
                    image = Image.open(io.BytesIO(image_data))
                    return image
        except Exception as e:
            print(f"Error loading image: {str(e)}")
            return None
