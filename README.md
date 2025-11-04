# GPT-2 Prefill阶段多精度误差分析实验

## 实验目标

本实验旨在分析GPT-2模型在不同精度（FP32, FP16, BF16等）下进行prefill推理时，KV cache和Attention kernel的误差累积情况。

## 功能特性

1. ✅ 使用Hugging Face部署GPT-2模型进行推理
2. ✅ 自动检测并使用GPU（如果可用）
3. ✅ 仅进行prefill阶段推理（不进行decode）
4. ✅ 生成20个句子，每句正好20个token
5. ✅ 支持多种精度：FP32（参考）、FP16、BF16
6. ✅ 统计并可视化每个token位置上的KV cache和Attention kernel误差变化

## 安装依赖

```bash
pip install -r requirements.txt
```

## 运行实验

### 方法1：使用脚本
```bash
./run_experiment.sh
```

### 方法2：直接运行
```bash
python experiment.py
```

## 输出结果

实验完成后会生成以下文件：

1. **sentences.json**: 生成的20个测试句子（每句20个token）
2. **plots/error_analysis_FP16.jpg**: FP16精度下的误差分析图
3. **plots/error_analysis_BF16.jpg**: BF16精度下的误差分析图（如果支持）

每个图表包含两个子图：
- **KV Cache误差图**: 显示每个token位置上的KV cache绝对误差
- **Attention Kernel误差图**: 显示每个token位置上的Attention权重绝对误差

## 实验原理

1. **参考精度**: 使用FP32作为参考精度（误差为0）
2. **测试精度**: 对FP16、BF16等精度进行测试
3. **误差计算**: 计算每个token位置上的绝对误差和相对误差
4. **误差聚合**: 对所有20个句子和所有层的误差进行聚合统计
5. **可视化**: 生成误差随token位置变化的图表

## 注意事项

- 首次运行会下载GPT-2模型（约500MB）
- 如果系统不支持BF16，将自动跳过BF16精度测试
- 实验需要一定时间完成（取决于硬件配置）
- 确保有足够的磁盘空间存储模型和结果

# tmptest1
# tmptest1
