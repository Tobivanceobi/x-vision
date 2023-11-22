import os

import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from skopt import BayesSearchCV
from xgboost import XGBClassifier

from dataLoader import DataLoader

ImageFile.LOAD_TRUNCATED_IMAGES = True

data_config = {
    # "Lung Lesion": 1000,
    # "Edema": 1000,
    "No Finding": 1000,
    "Consolidation": 1000,
    "Pneumonia": 1000,
    # "Atelectasis": 1000,
    # "Pneumothorax": 1000
}

dl = DataLoader()
dl.get_image_paths(data_config)
x_train, y_train = dl.load_images_from_files()


label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
print(label_encoder.get_params())

print(x_train.shape)
print(y_train.shape)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)

print(x_train.shape)
print(y_train.shape)

print(np.unique(y_train, return_counts=True))

skf_vals = []
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=126)
for fold, (train_index, test_index) in enumerate(skf.split(x_train, y_train)):
    skf_vals.append((train_index, test_index))

parameter_space = {
    'n_estimators': [1000],
    'learning_rate': [0.001, 0.1],
    'max_depth': [2, 4],
    'subsample': [0.4, 0.9],
    'colsample_bytree': [0.4, 1],
    'reg_lambda': [1, 15],
    'reg_alpha': [0, 10],
    'gamma': [0.1, 1]
}
model = XGBClassifier(
    n_jobs=-1,
    seed=27,
    tree_method="hist",
    device="cuda",
)

fit_param = {
    'early_stopping_rounds': 200,
}

clf = BayesSearchCV(estimator=model,
                    search_spaces=parameter_space,
                    fit_params=fit_param,
                    cv=skf_vals,
                    n_iter=20,
                    scoring='accuracy',
                    verbose=4)
print("starting training")
clf.fit(x_train, y=y_train)

print(clf.cv_results_)
print(clf.best_score_)
print(clf.best_params_)
results = pd.DataFrame(clf.cv_results_)
results.to_csv('./XGBoost_results.csv')
