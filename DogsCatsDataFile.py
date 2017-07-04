import numpy as np
import pandas as pd
from glob import glob
from settings import *
from cscreendataset import CommonDataSet

img_paths = glob(DATA_DIR + "/train/*.jpg")
data = []
for img_path in img_paths:
    img_name = img_path.split("/")[-1]
    name_parts = img_name.split(".")
    cat_or_dog = name_parts[0]
    img_id = img_name[0:img_name.index(".jpg")]
    val = [img_id, 1, 0]
    if cat_or_dog == "cat":
        val[1] = 0
        val[2] = 1
    data.append(val)

np_data = np.array(data)
np_data = np.random.permutation(np_data)

print(np_data.shape)
print(np_data[:5])

df = pd.DataFrame(data=np_data, columns=np.array(["id", "dog", "cat"]))

df.to_csv(DATA_DIR + '/train_labels.csv', index=False)

dset = CommonDataSet(DATA_DIR + '/train_labels.csv',
                     transform=None)
