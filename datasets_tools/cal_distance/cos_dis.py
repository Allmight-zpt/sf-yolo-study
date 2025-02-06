import os
import numpy as np
from tqdm import tqdm
import torch
from torchvision import models, transforms
from torch.nn import CosineSimilarity
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt


def calculate_feature_similarity(image1_path, image2_path):
    """
    使用Inception V3模型计算两张图片的特征向量，并返回它们之间的余弦相似度。

    :param image1_path: 第一张图片的路径。
    :param image2_path: 第二张图片的路径。
    :return: 两张图片特征向量之间的余弦相似度。
    """
    # 加载预训练的Inception V3模型（不包括顶部的全连接层）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inception_v3 = models.inception_v3(pretrained=True)
    inception_v3.aux_logits = False  # 禁用辅助输出
    inception_v3.fc = torch.nn.Identity()  # 替换最后的全连接层为恒等映射
    inception_v3 = inception_v3.to(device)
    inception_v3.eval()

    preprocess = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def load_and_preprocess_image(img_path):
        img = Image.open(img_path).convert('RGB')
        img = preprocess(img)
        return img.unsqueeze(0).to(device)

    image1 = load_and_preprocess_image(image1_path)
    image2 = load_and_preprocess_image(image2_path)

    with torch.no_grad():
        features1 = inception_v3(image1)
        features2 = inception_v3(image2)

    cos_sim = CosineSimilarity(dim=1)
    similarity = cos_sim(features1, features2).item()
    return 1 - similarity  # 将余弦相似度转换为距离形式


def normalize_data(data):
    """
    归一化数据到0-1范围。

    :param data: 需要归一化的数据列表或数组。
    :return: 归一化后的数据。
    """
    min_val = np.min(data)
    max_val = np.max(data)
    return (data - min_val) / (max_val - min_val)


def process_folders(origin_folder_paths, compared_folder_paths):
    """
    遍历多个原始文件夹和对应的比较文件夹，计算每张图片的特征相似度，并按文件夹分类整理数据。

    :param origin_folder_paths: 原始图片所在文件夹路径的列表。
    :param compared_folder_paths: 要与原始图片比较的图片所在文件夹路径的列表。
    :return: 各文件夹对应的特征相似度值列表。
    """
    folder_similarity_values = {}  # 存储每个文件夹的相似度值列表
    all_similarity_values = []  # 存储所有图片的相似度值，用于统一归一化

    for origin_folder, compared_folder in zip(tqdm(origin_folder_paths, desc='Processing folders', unit='folder'),
                                              compared_folder_paths):
        if not (os.path.isdir(origin_folder) and os.path.isdir(compared_folder)):
            print(f"路径 {origin_folder} 或 {compared_folder} 不是一个有效的文件夹!")
            continue

        similarity_values = []
        origin_files = [f for f in os.listdir(origin_folder) if
                        os.path.isfile(os.path.join(origin_folder, f)) and (f.endswith('.png') or f.endswith('.jpg'))]
        compared_files = [f for f in os.listdir(compared_folder) if
                          os.path.isfile(os.path.join(compared_folder, f)) and (
                                      f.endswith('.png') or f.endswith('.jpg'))]

        for origin_file, compared_file in tqdm(zip(origin_files, compared_files),
                                               desc=f'Processing images in {os.path.basename(origin_folder)}',
                                               unit='img', leave=False):
            origin_path = os.path.join(origin_folder, origin_file)
            compared_path = os.path.join(compared_folder, compared_file)
            similarity_value = calculate_feature_similarity(origin_path, compared_path)
            similarity_values.append(similarity_value)
            all_similarity_values.append(similarity_value)  # 将所有数据加入统一归一化列表

        if similarity_values:
            folder_similarity_values[origin_folder] = similarity_values

    # 对所有数据进行归一化
    all_similarity_values = np.array(all_similarity_values)
    normalized_all_similarities = normalize_data(all_similarity_values)

    # 将归一化后的数据重新赋值给各文件夹
    idx = 0
    for folder, ratios in folder_similarity_values.items():
        num_images = len(ratios)
        folder_similarity_values[folder] = list(normalized_all_similarities[idx: idx + num_images])
        idx += num_images

    return folder_similarity_values


def plot_heatmap(folder_compression_ratios, labels, num_bins=10):
    """
    绘制热力图，每行表示一个类别，横轴是压缩比率的分布区间，颜色表示数据量。

    :param folder_compression_ratios: 每个文件夹对应的压缩比列表
    :param labels: 类别标签（文件夹对应的名称）
    :param num_bins: 将压缩比率分成多少个区间
    """
    # 获取所有压缩比数据，确定统一的横轴范围
    all_ratios = []
    for ratios in folder_compression_ratios.values():
        all_ratios.extend(ratios)
    min_val, max_val = min(all_ratios), max(all_ratios)

    # 生成区间边界
    bins = np.linspace(min_val, max_val, num_bins + 1)

    # 统计每个类别在各区间的数据量
    heatmap_data = []
    for folder, ratios in folder_compression_ratios.items():
        hist, _ = np.histogram(ratios, bins=bins)
        heatmap_data.append(hist)

    heatmap_data = np.array(heatmap_data)  # 转为数组便于绘图

    # 绘制热力图
    plt.figure(figsize=(12, 6))
    plt.imshow(heatmap_data, aspect='auto', cmap='YlOrRd')

    # 设置横轴和纵轴标签
    plt.xticks(ticks=np.arange(num_bins), labels=[f'{bins[i]:.2f}' for i in range(num_bins)], rotation=45, fontdict={'fontsize': 18})
    plt.yticks(ticks=np.arange(len(labels)), labels=labels, fontdict={'fontsize': 18})

    plt.colorbar(label='Number of Images')
    # plt.title('Heatmap of Image Compression Ratios')
    # plt.xlabel('Compression Ratio Bins')
    # plt.ylabel('Categories')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':

    # 示例使用
    aug_folder_paths = [
        "./runs/train/exp/enhance_style_samples",
        "./runs/train/exp2/enhance_style_samples",
        "./runs/train/exp3/enhance_style_samples",
        "./runs/train/exp4/enhance_style_samples",
        "./runs/train/exp5/enhance_style_samples"
    ]

    origin_paths = [
        "./resize_origin",
        "./resize_origin",
        "./resize_origin",
        "./resize_origin",
        "./resize_origin"
    ]

    labels = [
        "alpha: 0.01",
        "alpha: 0.02",
        "alpha: 0.03",
        "alpha: 0.04",
        "alpha: 0.05"
    ]

    '''
    CAL：是否重新计算compress值并写入文件
    FROM_FILE:直接从文件中读取compress值
    PLOT:是否画图
    '''
    CAL = False
    From_FILE = True
    PLOT = True

    folder_compression_ratios = None
    if CAL:
        folder_compression_ratios = process_folders(aug_folder_paths, origin_paths)
        df = pd.DataFrame(folder_compression_ratios)
        df.to_csv('cos_ratios.csv', index=False)  # index=False 防止将行索引写入文件
    if From_FILE:
        folder_compression_ratios = pd.read_csv('cos_ratios.csv').iloc[:, 0:-1].to_dict(orient='list')
    if PLOT:
        plot_heatmap(folder_compression_ratios, labels, num_bins=15)
