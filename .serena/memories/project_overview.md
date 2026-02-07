LaFTer-masterTEXT 项目分析：

## 项目概述
LaFTer (Label-Free Tuning of Zero-shot Classifier) 是一个基于视觉-语言模型的零样本分类框架，发表在NeurIPS 2023。项目使用CLIP的共享图像-文本嵌入空间，通过文本描述训练分类器，然后应用于视觉数据。

## 技术栈
- Python 3.8.8, PyTorch 1.13.1, CUDA 11.1
- CLIP视觉-语言模型
- dassl框架 (基于CoOp)
- WandB用于实验监控
- 支持多种ViT架构 (B/16, B/32, L/14)

## 标签获取方式分析
项目有两种不同的标签获取模式：

### 1. 标准监督学习数据集 (如DTD, OxfordPets等)
- 使用真实标注文件 (如trainval.txt, test.txt)
- 通过Datum类存储和提供真实标签
- 完全监督的训练方式

### 2. 情感数据集 (Emotion6, FI, Emoset) 
- Emotion6.py: 从JSON文件获取真实情感标签 ("true_emotion"字段)
- 具有完整的标注信息和类别映射
- 本质上仍然是监督学习

### 3. 零样本模式 (zero_shot)
- 不使用任何真实标签进行训练
- 仅依赖CLIP的预训练能力进行推理

## 监督学习方式
- txt_cls参数控制训练模式: 'lafter', 'cls_only', 'templates_only', 'zero_shot'
- 除zero_shot模式外，其他模式都使用真实标签
- 数据集通过DatasetBase基类提供标准PyTorch Dataset接口
- 训练过程中可以访问真实的ground truth标签