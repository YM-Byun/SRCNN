import os
import shutil


src_path = './train'
dst_path = './val'

file_list = os.listdir(os.path.join(src_path, "hr"))

val_size = int(len(file_list) * 0.2)

for idx, image in enumerate(file_list):
    print (image)
    shutil.move(os.path.join(src_path, "hr", image), os.path.join(dst_path, "hr", image))
    shutil.move(os.path.join(src_path, "lr", image), os.path.join(dst_path, "lr", image))

    if idx == val_size:
        break