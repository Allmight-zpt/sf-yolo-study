import os
import shutil

# 文件路径
image_dir = './foggy_cityscapes_beta_0.01_voc_format/VOC2007/JPEGImages'
label_dir = './foggy_cityscapes_beta_0.01_voc_format/VOC2007/labels'
test_txt = './foggy_cityscapes_beta_0.01_voc_format/VOC2007/ImageSets/Main/test.txt'
trainval_txt = './foggy_cityscapes_beta_0.01_voc_format/VOC2007/ImageSets/Main/trainval.txt'

# 创建子文件夹
os.makedirs(os.path.join(image_dir, 'test'), exist_ok=True)
os.makedirs(os.path.join(image_dir, 'train'), exist_ok=True)
os.makedirs(os.path.join(label_dir, 'test'), exist_ok=True)
os.makedirs(os.path.join(label_dir, 'train'), exist_ok=True)

def move_files(file_list, src_dir, dst_dir, ext):
    for file_name in file_list:
        src_path = os.path.join(src_dir, f"{file_name}{ext}")
        dst_path = os.path.join(dst_dir, f"{file_name}{ext}")
        if os.path.exists(src_path):
            shutil.move(src_path, dst_path)

# 读取文件列表
with open(test_txt, 'r') as f:
    test_files = [line.strip() for line in f.readlines()]

with open(trainval_txt, 'r') as f:
    train_files = [line.strip() for line in f.readlines()]

# 移动图像文件
move_files(test_files, image_dir, os.path.join(image_dir, 'test'), '.png')
move_files(train_files, image_dir, os.path.join(image_dir, 'train'), '.png')

# 移动标签文件
move_files(test_files, label_dir, os.path.join(label_dir, 'test'), '.txt')
move_files(train_files, label_dir, os.path.join(label_dir, 'train'), '.txt')

print("文件已成功分类并移动到对应的子文件夹！")
