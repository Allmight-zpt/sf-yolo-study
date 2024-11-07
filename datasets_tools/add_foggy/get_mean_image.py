from PIL import Image
import numpy as np
import os

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        try:
            img_path = os.path.join(folder, filename)
            img = Image.open(img_path)
            # 确保所有图像具有相同的尺寸
            img = img.resize((2048, 1024))
            images.append(np.array(img))
        except (IOError, SyntaxError) as e:
            print(f"Bad file: {img_path} - {e}")
    return np.array(images)

def calculate_average_image(images):
    return np.mean(images, axis=0).astype(np.uint8)

folder_path = r'D:\file\work\pycharmFile\sf-yolo\datasets_tools\convert\leftImg8bit_foggy\train\aachen'
images = load_images_from_folder(folder_path)
if len(images) > 0:
    average_image = calculate_average_image(images)
    # 保存平均图像
    Image.fromarray(average_image).save('./average_foggy_image.jpg')
else:
    print("No valid images found in the folder.")