import numpy as np

with open("class_count_cityscapes_pixel.npy", "rb") as f:
    data = np.load(f)
data = data[:-1]
classes = ('road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign', 'vegetation',
           'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle')
print([(classes[i], data[i],i) for i in np.argsort(data)])
