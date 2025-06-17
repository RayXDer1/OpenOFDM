# OpenOFDM
# OFDM 基带物理层开源仿真项目

## 文件夹结构
- `DATA`：存储仿真和临时数据集
- `OFDM_TX`：OFDM发送基带子程序
- `OFDM_RX`：OFDM接收基带子程序
- `DEEP_LEARN1`：基于时域OFDM信号降噪的深度学习方法
- `DEEP_LEARN2`：基于频域OFDM信道估计值CSI降噪的深度学习方法
- `PMF-FFT`：PMF-FFT同步方法
- `DEEP_LEARN3`：高速移动场景下基于频域OFDM信号降噪的深度学习方法

## 主函数列表
- `main.m`：OFDM基带物理层主程序
- `train_test.m`：神经网络训练函数
- `main_dl_time_test1`：OFDM基带物理层深度学习方法DEEP_LEARN1的主程序
- `main_dl_time_test2`：OFDM基带物理层深度学习方法DEEP_LEARN2的主程序

## 开源项目信息
- **单位**：西安电子科技大学SCL实验室
- **开源人员**：Yang Qinghai、Ran Jing、Le Chi
- **实验环境**：Windows 11 64bit、NVIDIA GeForce RTX 3050 Laptop GPU
- **仿真平台及所需工具**： MATLAB R2023b、DeepLearning Tool Box
