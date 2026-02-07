import pandas as pd
import os
import torch
from typing import Dict, Optional, List
import time

class TextSupervisor:
    """
    文本监督器：负责加载CSV并匹配图片路径到文本描述
    实现SimEmotion的losstextimg损失计算
    """

    def __init__(self, csv_path: str, alpha: float = 0.3, dataset_name: str = None):
        """
        初始化文本监督器

        Args:
            csv_path: CSV文件路径
            alpha: 文本监督损失权重
            dataset_name: 数据集名称，用于路径匹配。如果为None，则从csv_path自动推断
        """
        self.csv_path = csv_path
        self.alpha = alpha
        self.dataset_name = dataset_name or self._infer_dataset_from_path(csv_path)
        self.image_text_map = {}
        self._load_and_process_csv()

    def _load_and_process_csv(self):
        """加载CSV并构建映射关系"""
        print(f"Loading text supervision data from: {self.csv_path}")
        start_time = time.time()

        try:
            df = pd.read_csv(self.csv_path)
            print(f"Loaded {len(df)} text-image pairs from CSV")

            # 构建相对路径映射关系
            for idx, row in df.iterrows():
                img_path = row['Image Path']
                caption = row['Caption']

                # 标准化路径
                normalized_path = img_path.replace('\\', '/')

                # 构建相对路径映射（去掉data/前缀）
                Dataset = self.dataset_name
                data_prefix = f'data/{Dataset}/'
                if normalized_path.startswith(data_prefix):
                    relative_path = normalized_path.replace(data_prefix, '')
                    self.image_text_map[relative_path] = caption

        except Exception as e:
            print(f"Error loading CSV: {e}")
            self.image_text_map = {}

        load_time = time.time() - start_time
        print(f"CSV loading completed in {load_time:.2f} seconds")
        print(f"Built mappings for {len(self.image_text_map)} relative paths")

    def get_text_for_image_path(self, img_path: str) -> Optional[str]:
        """
        根据图片完整路径获取文本描述

        Args:
            img_path: 图片的完整路径，如 'D:/.../Emotion6/anger/154.jpg'

        Returns:
            对应的文本描述，如果未找到返回None
        """
        if not img_path:
            return None

        # 标准化输入路径
        normalized_path = img_path.replace('\\', '/')

        # 使用相对路径匹配策略
        Dataset = self.dataset_name
        if f'{Dataset}/' in normalized_path:
            relative_path = normalized_path.split(f'{Dataset}/')[-1]
            if relative_path in self.image_text_map:
                return self.image_text_map[relative_path]

        return None


    def validate_coverage(self, image_paths: List[str]) -> Dict[str, float]:
        """
        验证文本覆盖率

        Args:
            image_paths: 图片路径列表

        Returns:
            包含覆盖统计的字典
        """
        total_images = len(image_paths)
        matched_images = 0
        unmatched_examples = []

        for img_path in image_paths:
            if self.get_text_for_image_path(img_path) is not None:
                matched_images += 1
            else:
                unmatched_examples.append(os.path.basename(img_path))

        coverage_rate = matched_images / total_images if total_images > 0 else 0

        print(f"Text Coverage Analysis:")
        print(f"  Total images: {total_images}")
        print(f"  Matched with text: {matched_images}")
        print(f"  Coverage rate: {coverage_rate:.2%}")

        if unmatched_examples and len(unmatched_examples) <= 10:
            print(f"  Unmatched examples: {unmatched_examples}")
        elif unmatched_examples:
            print(f"  First 10 unmatched examples: {unmatched_examples[:10]}")

        return {
            'total_images': total_images,
            'matched_images': matched_images,
            'coverage_rate': coverage_rate,
            'unmatched_examples': unmatched_examples
        }

    def compute_text_img_loss(self, model, images: torch.Tensor, texts: List[str]) -> torch.Tensor:
        """
        计算文本-图像对齐损失 (losstextimg)
        遵循SimEmotion的损失公式：loss = mean(1 - cosine_similarity)

        Args:
            model: LaFTerUFT模型
            images: 图像张量 [batch_size, channels, height, width]
            texts: 对应的文本描述列表

        Returns:
            文本-图像损失值
        """

        # 过滤掉无效的文本
        valid_indices = [i for i, text in enumerate(texts) if text is not None]

        if not valid_indices:
            return torch.tensor(0.0, device=images.device)

        # 只处理有效的图像-文本对
        valid_images = images[valid_indices]
        valid_texts = [texts[i] for i in valid_indices]

        try:
            # 直接使用Model的encode_image方法计算图像特征（保持梯度）
            # 修复：不使用model.image_features()因为它包含torch.no_grad()
            image_features = model.model.encode_image(valid_images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # 使用Model计算文本特征（冻结）
            from clip import tokenize
            with torch.no_grad():
                text_tokens = tokenize(valid_texts).to(valid_images.device)
                text_features = model.model.encode_text(text_tokens)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # 计算余弦相似度损失
            # losstextimg = mean(1 - cosine_similarity(image_features, text_features))
            similarity = torch.cosine_similarity(image_features, text_features, dim=-1)
            loss = torch.mean(1 - similarity)

            return loss

        except Exception as e:
            print(f"Error computing text-image loss: {e}")
            return torch.tensor(0.0, device=images.device)

def enhance_batch_with_text(batch, text_supervisor: TextSupervisor, model=None) -> tuple:
    """
    为批次数据添加文本描述

    Args:
        batch: 原始批次数据
        text_supervisor: 文本监督器实例
        model: 模型实例（用于设备信息）

    Returns:
        (images, labels, texts) 元组
    """
    # 现有的批次处理逻辑保持不变
    input = batch["img"]
    labels = batch["label"]
    input = torch.stack(input)

    # 获取设备信息
    if model is not None:
        device = model.device
    else:
        device = input.device if hasattr(input, 'device') else 'cuda'

    input = input.to(device)
    labels = labels.to(device)

    # 获取对应的文本描述
    texts = []

    # 方案1：直接从batch中获取impath键（与print_image_text_pairs保持一致）
    image_paths = batch.get("impath", [])

    # 如果直接获取失败，尝试其他方法
    if not image_paths:
        # 检查是否有image_paths属性
        if hasattr(batch, 'image_paths'):
            image_paths = batch.image_paths
        # 检查是否有data_items属性（Datum对象列表）
        elif hasattr(batch, 'data_items'):
            image_paths = [item.impath for item in batch.data_items]
        # 检查batch字典中是否有data_items键
        elif 'data_items' in batch:
            image_paths = [item.impath for item in batch['data_items']]
        # 检查batch字典中是否有image_paths键
        elif 'image_paths' in batch:
            image_paths = batch['image_paths']

    # 如果找到了图像路径，尝试获取文本
    if image_paths:
        for img_path in image_paths:
            text = text_supervisor.get_text_for_image_path(img_path)
            texts.append(text)
    else:
        # 备用方案：使用占位符
        batch_size = input.size(0)
        texts = [None] * batch_size
        if batch_size > 0 and batch_size <= 3:  # 只在前几个批次显示警告
            print(f"Warning: No image path information found in batch (size: {batch_size}), using None for text supervision")

    return input, labels, texts