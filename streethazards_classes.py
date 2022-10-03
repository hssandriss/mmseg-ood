import numpy as np

with open("class_count_street_hazards_pixel.npy", "rb") as f:
    data = np.load(f)
data = data[:-1]
classes = (
    # "unlabeled",
    'building',
    'fence',
    'other',
    'pedestrian',
    'pole',
    'road line',
    'road',
    'sidewalk',
    'vegetation',
    'car',
    'wall',
    'trafic sign',
    # "anomaly"
)
print([(classes[i], data[i], i) for i in np.argsort(data)])
