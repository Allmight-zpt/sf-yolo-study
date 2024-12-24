import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from pathlib import Path
import argparse
from TargetAugment.enhance_vgg16 import enhance_vgg16
from utils.general import increment_path
from torchvision import transforms
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, help='data path')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=960, help='train, val image size (pixels)')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--seed', type=int, default=0, help='Global training seed')
    parser.add_argument('--project', default='./runs/train', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    # Logger arguments
    parser.add_argument('--decoder_path', type=str, help='Decoder path')
    parser.add_argument('--encoder_path', type=str, help='Encoder path')
    parser.add_argument('--fc1', type=str, help='fc1 path')
    parser.add_argument('--fc2', type=str, help='fc2 path')
    parser.add_argument('--style_path', type=str, default="",
                        help='Path to the style image, if not specified, random style will be used')
    parser.add_argument('--style_add_alpha', type=float, default=1,
                        help='The amount of style to add to the image (between 0 and 1)')
    parser.add_argument('--save_style_samples', action='store_true', help='Save style samples images (useful to debug)')
    return parser.parse_known_args()[0] if known else parser.parse_args()


# 可视化部分图像
def show_images(data_loader, num_images=4):
    images, labels = next(iter(data_loader))  # 获取一个batch的图片和标签
    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
    for i in range(num_images):
        axes[i].imshow(images[i].permute(1, 2, 0) / 255)  # 转换为(H, W, C)格式
        axes[i].set_title(dataset.classes[labels[i]])  # 显示对应类别
        axes[i].axis('off')  # 关闭坐标轴
    plt.show()


class ImageDatasetWithPaths(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        # 获取所有图片文件的路径列表
        self.image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                            if os.path.isfile(os.path.join(folder_path, f)) and
                            (f.lower().endswith('.png') or f.lower().endswith('.jpg') or f.lower().endswith('.jpeg'))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        # 加载图像
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        # 返回图像张量和文件路径
        return image, image_path.split("\\")[-1]


if __name__ == '__main__':
    opt = parse_opt()
    opt.random_style = opt.style_path == ""
    opt.cuda = True
    opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
    opt.log_dir = Path(opt.save_dir) / './enhance_style_samples'
    dataset_path = opt.data  # 数据集文件夹路径
    adain = enhance_vgg16(opt)

    # 定义 transform，确保图像加载时为 [0, 255] 范围
    transform = transforms.Compose([
        transforms.Resize((opt.imgsz, opt.imgsz)),  # 如果需要 resize
        transforms.ToTensor(),  # 将图像转化为 [0, 1] 范围的 tensor
        transforms.Lambda(lambda x: x * 255),  # 将 [0, 1] 范围的图像值转为 [0, 255]
        # transforms.Normalize(mean=[0, 0, 0], std=[255, 255, 255])  # 确保均值和标准差为正确值，防止除法影响
    ])

    # 加载数据集
    dataset = ImageDatasetWithPaths(dataset_path, transform=transform)

    # 创建DataLoader
    data_loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False)

    # 可视化一些图像
    # show_images(data_loader)

    # 使用tqdm包装data_loader以创建进度条
    for images, paths in tqdm(data_loader, desc='Processing batches', unit='batch'):
        images = images.to(torch.float32).to('cuda', non_blocking=True)
        adain.add_style(images, 0, save_images=opt.save_style_samples, save_real_name=paths, style_only=True,
                        save_all=True)
