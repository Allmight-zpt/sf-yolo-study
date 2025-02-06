import os
from torchvision import transforms
from PIL import Image
from tqdm import tqdm


def resize_images_with_torch(input_folder, output_folder, size=(960, 960)):
    # 如果输出文件夹不存在，则创建它
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 定义转换操作
    transform = transforms.Compose([
        transforms.Resize(size),  # 调整大小
        transforms.ToTensor(),  # 将图片转换为Tensor，必要时进行归一化
        transforms.ToPILImage()  # 再将Tensor转换回PIL Image以保存
    ])

    # 获取所有图片文件，并过滤出有效的图片文件
    files = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f)) and
             f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

    # 使用tqdm包装迭代器，以便显示进度条
    for filename in tqdm(files, desc='Processing images', unit='img'):
        file_path = os.path.join(input_folder, filename)

        with Image.open(file_path) as img:
            # 如果图像是PNG并且具有alpha通道，则将其转换为RGB模式
            if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
                img = img.convert('RGB')

            # 应用变换
            resized_img = transform(img)

            # 保存到新文件夹，保持原文件名
            resized_img.save(os.path.join(output_folder, filename))


# 使用函数
resize_images_with_torch(r'D:\file\work\pycharmFile\sf-yolo\datasets_tools\datasets\origin_data', r'./resize_origin')
