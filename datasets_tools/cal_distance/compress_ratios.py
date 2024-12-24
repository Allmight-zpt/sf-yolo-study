from PIL import Image
import io
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm


def compress_png(input_path):
    """
    计算PNG图片压缩比，不保存图片

    :param input_path: 输入PNG文件路径
    :return: 压缩比
    """
    # 获取压缩前的文件大小
    original_size = os.path.getsize(input_path)

    # 打开图片
    img = Image.open(input_path)

    # 将图片保存到内存中（模拟压缩）
    compressed_image = io.BytesIO()
    img.save(compressed_image, format='PNG', compress_level=9)  # 使用Pillow进行压缩，compress_level=9表示最大压缩

    # 获取压缩后的文件大小（内存中的大小）
    compressed_size = len(compressed_image.getvalue())

    # 计算压缩比
    if original_size == 0:
        return 0

    # compression_ratio = original_size / compressed_size
    # 再加标准化与mood保持一致
    compression_ratio = compressed_size

    return compression_ratio


def normalize_data(data):
    """
    对数据进行Min-Max归一化，使数据在0到1之间

    :param data: 原始数据
    :return: 归一化后的数据
    """
    min_val = np.min(data)
    max_val = np.max(data)

    # 避免除以零
    if max_val - min_val == 0:
        return data

    return (data - min_val) / (max_val - min_val)


def get_file_paths(folder_name):
    ret = []
    if not os.path.isdir(folder_name):
        print(f"路径 {folder_name} 不是一个有效的文件夹!")
    for file_name in os.listdir(folder_name):
        file_path = os.path.join(folder_name, file_name)
        ret.append(file_path)
    return ret


def process_folders(folder_paths):
    """
    遍历多个文件夹，计算每张图片的压缩比，并按文件夹分类整理数据

    :param folder_paths: 文件夹路径的列表
    :return: 各文件夹对应的压缩比列表
    """
    folder_compression_ratios = {}  # 存储每个文件夹的压缩比列表
    all_compression_ratios = []  # 存储所有图片的压缩比，用于统一归一化

    # 使用tqdm包装folder_paths来为文件夹遍历创建进度条
    for folder in tqdm(folder_paths, desc='Processing folders', unit='folder'):
        if not os.path.isdir(folder):
            print(f"路径 {folder} 不是一个有效的文件夹!")
            continue

        compression_ratios = []
        # 获取文件夹内的所有文件，并使用tqdm包装以显示处理进度
        files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f)) and (f.endswith('.png') or f.endswith('.jpg'))]
        for filename in tqdm(files, desc=f'Processing images in {os.path.basename(folder)}', unit='img', leave=False):
            file_path = os.path.join(folder, filename)
            compression_ratio = compress_png(file_path)
            compression_ratios.append(compression_ratio)
            all_compression_ratios.append(compression_ratio)  # 将所有数据加入统一归一化列表

        if compression_ratios:
            folder_compression_ratios[folder] = compression_ratios

    # 对所有数据进行归一化
    all_compression_ratios = np.array(all_compression_ratios)
    normalized_all_ratios = normalize_data(all_compression_ratios)

    # 将归一化后的数据重新赋值给各文件夹
    idx = 0
    for folder, ratios in folder_compression_ratios.items():
        num_images = len(ratios)
        folder_compression_ratios[folder] = list(normalized_all_ratios[idx: idx + num_images])
        idx += num_images

    return folder_compression_ratios


def visualize_compression_ratios(folder_compression_ratios, labels):
    """
    可视化每个文件夹的压缩比数据

    :param folder_compression_ratios: 每个文件夹的压缩比列表
    :param labels: 每个文件夹的标签，用于图例
    """
    plt.figure(figsize=(10, 6))

    for idx, (_, ratios) in enumerate(folder_compression_ratios.items()):
        sns.histplot(ratios, bins=20, kde=True, stat="density", alpha=0.7, label=labels[idx])

    plt.title('PNG Image Compression Ratios')
    plt.xlabel('Compression Ratio')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.show()


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
    plt.xticks(ticks=np.arange(num_bins), labels=[f'{bins[i]:.2f}' for i in range(num_bins)], rotation=45)
    plt.yticks(ticks=np.arange(len(labels)), labels=labels)

    plt.colorbar(label='Number of Images')
    plt.title('Heatmap of Image Compression Ratios')
    plt.xlabel('Compression Ratio Bins')
    plt.ylabel('Categories')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    """
        # 示例使用
    aug_folder_paths = [
        "./runs/train/exp/enhance_style_samples",
        "./runs/train/exp2/enhance_style_samples",
        "./runs/train/exp3/enhance_style_samples",
        "./runs/train/exp4/enhance_style_samples",
        "./runs/train/exp5/enhance_style_samples"
    ]

    de_aug_folder_paths = [
        "./runs/train/exp6/enhance_style_samples",
        "./runs/train/exp7/enhance_style_samples",
        "./runs/train/exp8/enhance_style_samples",
        "./runs/train/exp9/enhance_style_samples",
        "./runs/train/exp10/enhance_style_samples"
    ]

    aug_de_aug_02_folder_paths = [
        "./runs/train/exp11/enhance_style_samples",
        "./runs/train/exp12/enhance_style_samples",
        "./runs/train/exp13/enhance_style_samples",
        "./runs/train/exp14/enhance_style_samples",
        "./runs/train/exp15/enhance_style_samples"
    ]

    aug_de_aug_ada_folder_paths = [
        "./runs/train/exp16/enhance_style_samples",
        "./runs/train/exp17/enhance_style_samples",
        "./runs/train/exp18/enhance_style_samples",
        "./runs/train/exp19/enhance_style_samples",
        "./runs/train/exp20/enhance_style_samples"
    ]

    aug_de_aug_cycle_folder_paths = [
        "./sub_datasets_foggy_0.2_cycle/origin_data",
        "./sub_datasets_foggy_0.4_cycle/origin_data",
        "./sub_datasets_foggy_0.6_cycle/origin_data",
        "./sub_datasets_foggy_0.7_cycle/origin_data",
        "./sub_datasets_foggy_0.8_cycle/origin_data"
    ]

    # cycle后统一加0.4
    aug_and_de_aug_cycle_folder_paths = [
        "./sub_datasets_foggy_0.2/origin_data",
        "./sub_datasets_foggy_0.4/origin_data",
        "./sub_datasets_foggy_0.6/origin_data",
        "./sub_datasets_foggy_0.7/origin_data",
        "./sub_datasets_foggy_0.8/origin_data",
        "./runs/train/exp25/enhance_style_samples",
        "./runs/train/exp24/enhance_style_samples",
        "./runs/train/exp23/enhance_style_samples",
        "./runs/train/exp22/enhance_style_samples",
        "./runs/train/exp21/enhance_style_samples",
        "./sub_datasets_foggy/origin_data"
    ]

    labels = [
        "alpha: 0.2",
        "alpha: 0.4",
        "alpha: 0.6",
        "alpha: 0.7",
        "alpha: 0.8"
    ]

    labels_all = [
        "alpha: 0.2",
        "alpha: 0.4",
        "alpha: 0.6",
        "alpha: 0.7",
        "alpha: 0.8",
        "alpha_cycle: 0.2",
        "alpha_cycle: 0.4",
        "alpha_cycle: 0.6",
        "alpha_cycle: 0.7",
        "alpha_cycle: 0.8",
        "foggy"
    ]
    """

    # 1. 数据源
    three_images = [
        "./sub_datasets_foggy_cycle/origin_data",
        "./sub_datasets_foggy/origin_data",
        "./sub_datasets_foggy_foggy_0.4/origin_data"
    ]
    # 2. 数据源标签
    three_labels = [
        "deaug",
        "origin",
        "aug"
    ]

    '''
    CAL：是否重新计算compress值并写入文件
    FROM_FILE:直接从文件中读取compress值
    PLOT:是否画图
    '''
    CAL = True
    From_FILE = False
    PLOT = True

    folder_compression_ratios = None
    if CAL:
        folder_compression_ratios = process_folders(three_images)
        df = pd.DataFrame(folder_compression_ratios)
        file_paths = get_file_paths("./sub_datasets_foggy/origin_data")
        df['file_paths'] = file_paths
        df.to_csv('compression_ratios.csv', index=False)  # index=False 防止将行索引写入文件
    if From_FILE:
        folder_compression_ratios = pd.read_csv('compression_ratios.csv').iloc[:, 0:-1].to_dict(orient='list')
    if PLOT:
        visualize_compression_ratios(folder_compression_ratios, three_labels)
        plot_heatmap(folder_compression_ratios, three_labels, num_bins=15)
