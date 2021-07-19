import os
from PIL import Image
import numpy as np

SUBIMAGE_SIZE = 100
STRIDE = 20
PADDING = 20

# Numpy array shape (h, w, c)
NUMPY_HEIGHT = 0
NUMPY_WIDHT = 1
NUMPY_CHANNEL = 2

origin_path = './origin'
dst_path = './train'

file_list = os.listdir(origin_path)

for image in file_list:
    origin_image = Image.open(os.path.join(origin_path, image))
    origin_arr = np.array(origin_image)

    start_x = PADDING
    start_y = PADDING

    while start_y + SUBIMAGE_SIZE < origin_arr[NUMPY_HEIGHT] + PADDING:

        sub_image = origin_arr[start_y:start_y+SUBIMAGE_SIZE, start_x:start_x+SUBIMAGE_SIZE, :]

        if start_x + STRIDE + SUBIMAGE_SIZE > origin_arr[NUMPY_WIDHT] + PADDING:
            start_x = PADDING
            start_y += STRIDE
        else:
            start_x += STRIDE