import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image


# 加载VGG16模型并分层
class VGG16FeatureExtractor(nn.Module):
    def __init__(self, pretrained_weights_path):
        super(VGG16FeatureExtractor, self).__init__()
        # 加载VGG16模型
        vgg = models.vgg16()
        vgg = nn.Sequential(*list(vgg.features._modules.values())[:-1])
        vgg[4].ceil_mode = True
        vgg[9].ceil_mode = True
        vgg[16].ceil_mode = True
        # 加载本地的预训练权重
        vgg.load_state_dict(torch.load(pretrained_weights_path)['model'])
        vgg = nn.Sequential(*list(vgg.children())[:19])
        # 获取VGG16的特征部分（卷积层）
        self.encoders = []
        self.encoders.append(nn.Sequential(*list(vgg._modules.values())[:2]))
        self.encoders.append(nn.Sequential(*list(vgg._modules.values())[2:7]))
        self.encoders.append(nn.Sequential(*list(vgg._modules.values())[7:12]))
        self.encoders.append(nn.Sequential(*list(vgg._modules.values())[12:]))

    def forward(self, x):
        features = [x]
        for i in range(4):
            features.append(self.encoders[i](features[-1]))
        return features[1:]


# 图像预处理函数
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((960, 960)),  # 调整大小到 960x960
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)  # 增加batch维度


# 全局平均池化
def global_avg_pooling(feature):
    return F.adaptive_avg_pool2d(feature, (1, 1)).view(feature.size(0), -1)


# 计算余弦距离
def cosine_distance(feature1, feature2):
    return 1 - F.cosine_similarity(feature1, feature2, dim=1)


# 主程序
def main():
    # 指定预训练权重路径和图片路径
    pretrained_weights_path = "../../TargetAugment_train/pre_trained/vgg16_ori.pth"
    origin_data = "../../datasets_tools/datasets/origin_data/aachen_000000_000019_leftImg8bit.png"
    foggy_datas = [
        "../../datasets_tools/datasets/foggy_data_0.01/aachen_000000_000019_leftImg8bit.png",
        "../../datasets_tools/datasets/foggy_data_0.02/aachen_000000_000019_leftImg8bit.png",
        "../../datasets_tools/datasets/foggy_data_0.04/aachen_000000_000019_leftImg8bit.png",
        "../../datasets_tools/datasets/foggy_data_0.06/aachen_000000_000019_leftImg8bit.png",
        "../../datasets_tools/datasets/foggy_data_0.08/aachen_000000_000019_leftImg8bit.png",
        "../../datasets_tools/datasets/foggy_data_0.1/aachen_000000_000019_leftImg8bit.png",
    ]

    # 加载模型
    model = VGG16FeatureExtractor(pretrained_weights_path)
    model.eval()  # 设置为评估模式

    # 读取和预处理图片
    origin_data = preprocess_image(origin_data)
    foggy_datas = [preprocess_image(data) for data in foggy_datas]

    # 确保输入尺寸正确
    print(f"Input image size: {origin_data.size()}")  # 应为 [1, 3, 960, 960]

    # 将图片输入模型，获取每一层的特征
    with torch.no_grad():
        features_origin = model(origin_data)
        features_foggy = [model(data) for data in foggy_datas]

    # 对每一层的特征进行池化并计算余弦距离
    for fea in features_foggy:
        print("*"*20)
        for i, (feat1, feat2) in enumerate(zip(features_origin, fea)):
            pooled_feat1 = global_avg_pooling(feat1)
            pooled_feat2 = global_avg_pooling(feat2)
            distance = cosine_distance(pooled_feat1, pooled_feat2)
            print(f"Layer {i + 1} cosine distance: {distance.item()}")


if __name__ == "__main__":
    main()
