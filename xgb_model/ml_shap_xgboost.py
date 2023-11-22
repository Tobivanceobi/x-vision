import random

import pandas as pd
import shap
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
import os
import numpy as np
from PIL import Image, ImageFile
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import matplotlib.colors as mcolors
ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_images_from_folder(base_dir, num_samples):
    image_data = []
    image_labels = []

    labels = ['NORMAL', 'PNEUMONIA']
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    for label, dir_name in zip(encoded_labels, labels):
        counter = 0
        print(f"loading data for {dir_name} labels")
        folder = os.path.join(base_dir, dir_name)
        for filename in os.listdir(folder):
            if counter == int(num_samples / 2):
                break
            # Construct full file path
            file_path = os.path.join(folder, filename)
            if file_path.lower().endswith(".jpeg"):
                # Open the image file
                with Image.open(file_path) as img:
                    # Convert to grayscale (if not already) and resize
                    img = img.convert('L').resize((100, 100))
                    # Convert image data to a numpy array and flatten it
                    img_array = np.array(img).flatten()
                    # Append the image data and label to the respective lists
                    image_data.append(img_array)
                    image_labels.append(label)
                    counter += 1
    return np.array(image_data), np.array(image_labels)


x, y = load_images_from_folder('data/train/', 5000)
x_val, y_val = load_images_from_folder('data/val/', 5000)
x_test, y_test = load_images_from_folder('data/test/', 5000)
# skf_vals = []
# skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=126)
# for fold, (train_index, test_index) in enumerate(skf.split(x, y)):
#     skf_vals.append((train_index, test_index))

res_df = pd.read_csv('./XGBoost_results.csv')
res_df = res_df.sort_values(by=['mean_test_score'], ascending=False)
best_params = res_df.iloc[0]
print(best_params)


model_param = dict()
for col in res_df.columns:
    if 'param_' in col:
        key_n = col.replace('param_', '')
        model_param[key_n] = best_params[col]
model_param['n_estimators'] = 3000
fold = 2
# x_train = [x[i] for i in skf_vals[fold][0]]
# x_test = [x[i] for i in skf_vals[fold][1]]
# y_train = [y[i] for i in skf_vals[fold][0]]
# y_test = [y[i] for i in skf_vals[fold][1]]
model = XGBClassifier(
    **model_param,
    n_jobs=-1,
    seed=27,
    tree_method="hist",
    device="cuda",
    early_stopping_rounds=150
)
model.fit(x, y=y, eval_set=[(x_test, y_test)])
preds = model.predict(x_val)
acc = accuracy_score(y_val, preds)
print(acc)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(x_val)

# shap_values_abs = np.abs(shap_values)
# mean_abs_shap_values = np.mean(shap_values_abs, axis=0)
print('image shap')
for i in range(0, 50):
    indx = random.randint(0, len(x_val)-1)
    shap_ix = shap_values[indx]
    print(f"Target: {y_val[indx]}, Pred: {preds[indx]}")
    # scaler = MinMaxScaler()
    # shap_ix = scaler.fit_transform(shap_ix.reshape(-1, 1))
    shap_ix = shap_ix.flatten()
    shap_image = shap_ix.reshape(100, 100)

    # Normalize SHAP values to be in the range 0-1
    # This step is important for visualization purposes
    max_val = np.max(np.abs(shap_image))
    shap_image_normalized = shap_image / max_val

    # Convert the original image to a PIL Image for display
    img_arr = x_val[indx].reshape(100, 100)
    original_image = Image.fromarray(img_arr.astype('uint8'), 'L')

    print('plotting figure')
    # Create a heatmap from the SHAP values
    plt.figure(figsize=(10, 5))

    # Show the original image
    plt.subplot(1, 2, 1)
    plt.imshow(original_image, cmap='gray')
    plt.axis('off')

    colors = [(0, 1, 0, 1),  # Green, opaque
              (0, 1, 0, 0),  # Green, transparent
              (1, 0, 0, 0),  # Red, transparent
              (1, 0, 0, 1)]  # Red, opaque
    locations = [0.0, 0.4, 0.6, 1.0]
    custom_colormap = mcolors.LinearSegmentedColormap.from_list('custom_gradient', list(zip(locations, colors)))

    # Show the SHAP values heatmap
    plt.subplot(1, 2, 2)
    plt.imshow(original_image, cmap='gray', alpha=0.7)  # Show the image
    plt.imshow(shap_image_normalized, cmap=custom_colormap, clim=(-0.5, 0.5), alpha=0.5)  # Overlay the SHAP heatmap
    plt.colorbar()
    plt.title("NORMAL=0; PNEUMONIA=1")

    plt.tight_layout()
    plt.show()
