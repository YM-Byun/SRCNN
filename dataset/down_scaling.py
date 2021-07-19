import os
import cv2

src_path = './train/hr'
dst_path = './train/lr'

file_list = os.listdir(src_path)

for idx, image in enumerate(file_list):
    val_img = cv2.imread(os.path.join(src_path, image), cv2.IMREAD_COLOR)

    small_img = cv2.resize(val_img, (80, 80), interpolation=cv2.INTER_CUBIC)

    big_img = cv2.resize(small_img, (240, 240), interpolation=cv2.INTER_CUBIC)

    cv2.imwrite(os.path.join(dst_path, image), big_img)

    print (f"{idx+ 1} / {len(file_list)}")
