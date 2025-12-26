# 1*1 convention 
import torch 
import torch.nn as nn

# 1*1 convolution for channel up/down sampling
input_torch = torch.randn(1,3,4,4)
conv1x1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=1)
output_torch = conv1x1(input_torch)
print("Input shape:", input_torch.shape)
print("Output shape from Conv2d:", output_torch.shape)
# 1*1 卷积用于减少计算量，并保存空间信息
# linear layer for channel up/down sampling
input_linear = input_torch
linear_layer = nn.Linear(in_features=3, out_features=6)
input_reshaped = input_linear.permute(0,2,3,1) # (batch, height, width, channels)
output_linear = linear_layer(input_reshaped)
output_linear = output_linear.permute(0,3,1,2) # back to (batch, channels, height, width)
print("Output shape from Conv2d:", output_torch.shape)
# 线性层用于最后聚合结果，但是会失去维度信息。