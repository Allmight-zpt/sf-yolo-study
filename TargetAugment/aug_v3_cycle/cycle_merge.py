import torch
import torch.nn as nn
import torch.optim as optim


# 定义一个包含1x1卷积的模型
class FullyConnectedLayer(nn.Module):
    def __init__(self):
        super(FullyConnectedLayer, self).__init__()
        # 1x1卷积层，将输入通道数从6变为3
        self.conv1x1 = nn.Conv2d(in_channels=6, out_channels=3, kernel_size=1)

    def forward(self, x):
        return self.conv1x1(x)

if __name__ == '__main__':
    # 模拟数据
    batch_size, in_channels, height, width = 16, 6, 960, 960
    input_data_1 = torch.randn(batch_size, 3, height, width)  # 输入数据
    input_data_2 = torch.randn(batch_size, 3, height, width)  # 输入数据
    input_data = torch.concat([input_data_1, input_data_2], dim=1)
    target_data = torch.randn(batch_size, 3, height, width)  # 目标数据 (假设输出为3通道)

    # 定义模型
    model = FullyConnectedLayer()

    # 定义损失函数
    criterion = nn.MSELoss()  # 使用均方误差作为损失函数

    # 定义优化器
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练循环
    num_epochs = 5
    for epoch in range(num_epochs):
        # 前向传播
        output = model(input_data)  # 计算输出
        loss = criterion(output, target_data)  # 计算损失

        # 反向传播
        optimizer.zero_grad()  # 清除梯度
        loss.backward()  # 计算梯度
        optimizer.step()  # 更新参数

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
