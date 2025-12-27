# nn.linear 应用在最后一个维度中
# 示例
import torch
import torch.nn as nn

# 输入张量形状 (batch_size, seq_length, feature_dim)
x = torch.randn(2, 3, 4)
linear = nn.Linear(in_features=4, out_features=5)
output = linear(x)
print("Input shape:", x.shape)
print("Output shape:", output.shape)

x_conv = torch.randn(4, 4, 3)  # (batch_size, in_channels, seq_length)
conv = nn.Conv1d(in_channels=4, out_channels=5, kernel_size=1)
x_conv_output = conv(x_conv)
print("Output shape from Conv1d:", x_conv_output.shape)
