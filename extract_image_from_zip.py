import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import shutil
import zipfile
print(os.listdir("./input"))
#################################

cache_dir = os.path.expanduser(os.path.join('~', '.keras'))
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

# Create symbolic links for trained models.
# Thanks to Lem Lordje Ko for the idea
# https://www.kaggle.com/lemonkoala/pretrained-keras-models-symlinked-not-copied
models_symlink = os.path.join(cache_dir, 'models')
if not os.path.exists(models_symlink):
    os.symlink('./input/keras_pretrained_models/', models_symlink)

images_dir = os.path.expanduser(os.path.join('~', 'avito_images'))
if not os.path.exists(images_dir):
    os.makedirs(images_dir)

#######################

NUM_IMAGES_TO_EXTRACT = 101

with zipfile.ZipFile('./input/data/train_jpg.zip', 'r') as train_zip:
    files_in_zip = sorted(train_zip.namelist())
    for idx, file in enumerate(files_in_zip[:NUM_IMAGES_TO_EXTRACT]):
        if file.endswith('.jpg'):
            train_zip.extract(file, path=file.split('/')[3])



