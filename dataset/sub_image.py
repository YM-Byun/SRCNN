import os
from PIL import Image
import numpy as np

SUBIMAGE_SIZE = 240
STRIDE = 200
PADDING = 100

# Numpy array shape (h, w, c)
NUMPY_HEIGHT = 0
NUMPY_WIDHT = 1
NUMPY_CHANNEL = 2

origin_path = './origin'
dst_path = './train/hr'

file_list = os.listdir(origin_path)

for idx, image in enumerate(file_list):
    origin_image = Image.open(os.path.join(origin_path, image))
    origin_arr = np.array(origin_image)

    start_x = PADDING
    start_y = PADDING

    cnt = 0

    while start_y + SUBIMAGE_SIZE < origin_arr.shape[NUMPY_HEIGHT] + PADDING:

        sub_image_arr = origin_arr[start_y:start_y+SUBIMAGE_SIZE, start_x:start_x+SUBIMAGE_SIZE, :]

        sub_image = Image.fromarray(sub_image_arr)
        sub_image.save(os.path.join(dst_path, image.replace(".jpg", "") + "_" + str(cnt) + ".jpg"))

        if start_x + STRIDE + SUBIMAGE_SIZE > origin_arr.shape[NUMPY_WIDHT] + PADDING:
            start_x = PADDING
            start_y += STRIDE
        else:
            start_x += STRIDE

        cnt += 1

    print (f"{idx+ 1} / {len(file_list)}")
