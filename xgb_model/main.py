import pandas as pd
import pandas as pd
import zipfile
from PIL import Image
import io

BASE_PATH = '/run/media/tobias/Archiv/CheXpert_Dataset/chexpertchestxrays-u20210408/'

data_config = {
    "Lung Lesion": 2000,
    "Edema": 2000,
    "Consolidation": 2000,
    "Pneumonia": 2000,
    "Atelectasis": 2000,
    "Pneumothorax": 2000
}

processing_config = {
    "resize"
}


def load_jpeg_from_zip(zip_file_path, image_filename):
    try:
        with zipfile.ZipFile(zip_file_path, 'r') as zip_file:
            with zip_file.open(image_filename) as image_file:
                image_data = image_file.read()
                image = Image.open(io.BytesIO(image_data))
                return image
    except Exception as e:
        print(f"Error loading image: {str(e)}")
        return None


if __name__ == '__main__':
    df = pd.read_csv(BASE_PATH + 'train_cheXbert.csv')
    pneumonia_rows = df[df['Pneumonia'] == 1]
    pneumothorax_rows = df[df['Pneumothorax'] == 1]
    atelectasis_rows = df[df['Atelectasis'] == 1]
    image_filename = atelectasis_rows.iloc[0]['Path']

    zip_file_path = BASE_PATH + 'CheXpert-v1.0.zip'

    # loaded_image = load_jpeg_from_zip(zip_file_path, image_filename)
    # if loaded_image:
    #     loaded_image.show()  # Display the loaded image using PIL

