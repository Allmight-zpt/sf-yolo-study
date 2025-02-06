import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import pandas as pd
from pathlib import Path
from TargetAugment.enhance_vgg16 import enhance_vgg16
from utils.general import increment_path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch
import argparse
from tqdm import tqdm


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=960, help='train, val image size (pixels)')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--seed', type=int, default=0, help='Global training seed')
    parser.add_argument('--project', default='./runs/train', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--compression_dis', default='compression_dis.csv', type=str, help='compression distance infomation path')
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


class CustomDataset(Dataset):
    def __init__(self, data_frame, transform=None):
        self.data_frame = data_frame
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_path = self.data_frame.iloc[idx, 2]  # 图片路径在第三列
        image = Image.open(img_path).convert('RGB')  # 读取图片并转换为 RGB 模式
        if self.transform:
            image = self.transform(image)  # 应用数据增强或预处理操作

        label = int(self.data_frame.iloc[idx, 3])  # 获取类别标签（最后一列）

        # 返回图片、标签和图片路径
        return image, label, img_path.split("\\")[-1]


def create_dataset_and_dataloader(category_df, transform, batch_size=16):
    if len(category_df) > 0:
        dataset = CustomDataset(category_df, transform=transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        return dataset, dataloader
    else:
        print("Warning: No data in this category.")
        return None, None


def adaptive_augment(adain, dataloader, flag, save_style_samples):
    if dataloader:
        if flag == 0:
            adain.alpha = 0.4
        elif flag == 1:
            adain.alpha = 0.4
        elif flag == 2:
            adain.alpha = 0.4
        elif flag == 3:
            adain.alpha = 0.4
        # 使用tqdm包装dataloader以创建进度条
        for images, _, paths in tqdm(dataloader, desc='Processing batches', unit='batch'):
            images = images.to(torch.float32).to('cuda', non_blocking=True)
            adain.add_style(images, 0, save_images=save_style_samples, save_real_name=paths, style_only=True,
                            save_all=True)
    else:
        print("No data to show for this category.")


if __name__ == '__main__':
    opt = parse_opt()
    opt.random_style = opt.style_path == ""
    opt.cuda = True
    opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
    opt.log_dir = Path(opt.save_dir) / './enhance_style_samples'
    adain = enhance_vgg16(opt)
    df = pd.read_csv(opt.compression_dis)

    # 根据 category 列进行四分类
    category_0 = df[df['category'] == 0]
    category_1 = df[df['category'] == 1]
    category_2 = df[df['category'] == 2]
    category_3 = df[df['category'] == 3]

    transform = transforms.Compose([
        transforms.Resize((opt.imgsz, opt.imgsz)),  # 如果需要 resize
        transforms.ToTensor(),  # 将图像转化为 [0, 1] 范围的 tensor
        transforms.Lambda(lambda x: x * 255),  # 将 [0, 1] 范围的图像值转为 [0, 255]
        # transforms.Normalize(mean=[0, 0, 0], std=[255, 255, 255])  # 确保均值和标准差为正确值，防止除法影响
    ])

    # 创建数据集和 DataLoader（跳过空类别）
    dataset_0, dataloader_0 = create_dataset_and_dataloader(category_0, transform)
    dataset_1, dataloader_1 = create_dataset_and_dataloader(category_1, transform)
    dataset_2, dataloader_2 = create_dataset_and_dataloader(category_2, transform)
    dataset_3, dataloader_3 = create_dataset_and_dataloader(category_3, transform)

    adaptive_augment(adain, dataloader_0, 0, opt.save_style_samples)
    adaptive_augment(adain, dataloader_1, 1, opt.save_style_samples)
    adaptive_augment(adain, dataloader_2, 2, opt.save_style_samples)
    adaptive_augment(adain, dataloader_3, 3, opt.save_style_samples)
