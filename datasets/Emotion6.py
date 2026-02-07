import os
import json
import random
import pickle
import numpy as np
import torch
from collections import defaultdict
from dassl.data.datasets import DATASET_REGISTRY, DatasetBase
from dassl.utils import listdir_nohidden, mkdir_if_missing
import math

# 基础的6个情绪类别，不包括 neutral
PRIMARY_EMOTIONS = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]
# 建立情绪名称到整数标签的映射
CLASSNAME_TO_LABEL = {name: i for i, name in enumerate(PRIMARY_EMOTIONS)}
LABEL_TO_CLASSNAME = {i: name for i, name in enumerate(PRIMARY_EMOTIONS)}
NUM_CLASSES = len(PRIMARY_EMOTIONS)


class Datum:
    def __init__(self, impath, label, domain, classname):
        self._impath = impath
        self._label = int(label)
        self._domain = int(domain)
        self._classname = str(classname)

    @property
    def impath(self):
        return self._impath

    @property
    def label(self):
        return self._label

    @property
    def domain(self):
        return self._domain

    @property
    def classname(self):
        return self._classname
    
    
@DATASET_REGISTRY.register()
class Emotion6(DatasetBase):
    dataset_dir = "Emotion6"  # 类属性，指定数据集文件夹名称
    domains = []  # 如果不需要特定域名，可以为空

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        # 正确构建数据集特定目录的路径
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        
        self.annotation_file = os.path.join(self.dataset_dir, "Emotion6.json")
        self.image_data_dir = self.dataset_dir
        
        # 创建用于保存拆分数据集信息的目录
        self.split_dir = os.path.join(self.dataset_dir, "split_info")
        mkdir_if_missing(self.split_dir)
        
                
        # 从JSON文件读取所有数据项
        all_items = self._read_data_from_json(self.annotation_file)
        
        # 获取数据划分策略配置
        split_strategy = getattr(cfg.DATASET, 'SPLIT_STRATEGY', 'stratified')  # 默认为随机划分
        
        # 根据不同策略进行数据集拆分 (在首次使用前固定随机种子，确保可复现)
        split_seed = getattr(cfg.DATASET, 'SPLIT_SEED', None)
        if split_seed is None:
            split_seed = getattr(cfg, 'SEED', 42) or 42
        random.seed(split_seed)

        if split_strategy.lower() == 'predefined':
            train_items, test_items = self._predefined_split(all_items, cfg)
        elif split_strategy.lower() == 'stratified':
            train_items, test_items = self._stratified_split(all_items, cfg)
        else:  # 默认随机划分
            train_items, test_items = self._random_split(all_items, cfg)
        
        # 处理少样本学习 (few-shot learning) 的情况
        num_shots = cfg.DATASET.NUM_SHOTS
        if num_shots >= 1 and len(train_items) > 0:
            seed = cfg.SEED
            preprocessed_fewshot_file = os.path.join(
                self.split_dir, f"shot_{num_shots}-seed_{seed}-{split_strategy}.pkl"
            )
            train_items = self._generate_fewshot_dataset(train_items, num_shots, seed)
        
        super().__init__(train_x=train_items, val=test_items, test=test_items)

    def _read_data_from_json(self, json_file):
        """从JSON文件读取数据项"""
        with open(json_file, 'r') as f:
            data = json.load(f)

        all_items = []
        print(f"Reading data from {json_file} located in {self.dataset_dir}")

        for key, value in data.items():
            true_emotion = value.get("true_emotion")

            if not true_emotion or true_emotion not in PRIMARY_EMOTIONS:
                continue

            impath = os.path.join(self.image_data_dir, key)

            if not os.path.exists(impath):
                print(f"Warning: Image file not found: {impath}. Skipping this item.")
                continue

            label = CLASSNAME_TO_LABEL[true_emotion]
            domain = 0
            classname = true_emotion

            item = Datum(impath=impath, label=label, domain=domain, classname=classname)
            all_items.append(item)

        if not all_items:
            raise ValueError(f"No data items were loaded from {json_file}.")

        print(f"Successfully loaded {len(all_items)} items from {json_file}.")
        return all_items

    def _random_split(self, all_items, cfg):
        """随机划分数据集"""
        print("Using random split strategy")
        random.shuffle(all_items)
        split_ratio = getattr(cfg.DATASET, 'SPLIT_RATIO', 0.8)
        split_idx = int(len(all_items) * split_ratio)
        
        train_items = all_items[:split_idx]
        test_items = all_items[split_idx:]
        
        print(f"Random split: {len(train_items)} train, {len(test_items)} test")
        return train_items, test_items

    def _stratified_split(self, all_items, cfg):
        """分层划分数据集（保持各类别比例）"""
        print("Using stratified split strategy")
        
        # 按类别分组
        items_by_class = defaultdict(list)
        for item in all_items:
            items_by_class[item.classname].append(item)
        
        train_items = []
        test_items = []
        split_ratio = getattr(cfg.DATASET, 'SPLIT_RATIO', 0.9)
        
        for classname, class_items in items_by_class.items():
            random.shuffle(class_items)
            split_idx = int(len(class_items) * split_ratio)
            
            train_items.extend(class_items[:split_idx])
            test_items.extend(class_items[split_idx:])
        
        # 重新随机化整体顺序
        random.shuffle(train_items)
        random.shuffle(test_items)
        
        print(f"Stratified split: {len(train_items)} train, {len(test_items)} test")
        return train_items, test_items

    def _predefined_split(self, all_items, cfg):
        """使用预定义的数据划分"""
        print("Using predefined split strategy")
        
        # 尝试多种预定义划分文件格式
        split_file_formats = [
            os.path.join(self.dataset_dir, "train_test_split.json"),
            os.path.join(self.dataset_dir, "splits", "train_test_split.json"),
            os.path.join(self.split_dir, "predefined_split.json")
        ]
        
        split_file = None
        for file_path in split_file_formats:
            if os.path.exists(file_path):
                split_file = file_path
                break
        
        if split_file is None:
            print("Warning: No predefined split file found. Falling back to random split.")
            return self._random_split(all_items, cfg)
        
        # 读取预定义划分
        with open(split_file, 'r') as f:
            split_data = json.load(f)
        
        # 创建文件名到项目的映射
        item_map = {}
        for item in all_items:
            # 提取相对于数据集目录的文件路径作为键
            rel_path = os.path.relpath(item.impath, self.dataset_dir)
            item_map[rel_path] = item
        
        train_items = []
        test_items = []
        
        # 根据预定义划分分配数据
        for file_path in split_data.get('train', []):
            if file_path in item_map:
                train_items.append(item_map[file_path])
        
        for file_path in split_data.get('test', []):
            if file_path in item_map:
                test_items.append(item_map[file_path])
        
        # 处理未在预定义划分中的项目
        used_files = set(split_data.get('train', [])) | set(split_data.get('test', []))
        unused_items = [item for rel_path, item in item_map.items() if rel_path not in used_files]
        
        if unused_items:
            print(f"Warning: {len(unused_items)} items not in predefined split. Adding to train set.")
            train_items.extend(unused_items)
        
        print(f"Predefined split: {len(train_items)} train, {len(test_items)} test")
        return train_items, test_items

    def _generate_fewshot_dataset(self, train_items, num_shots, seed):
        """生成少样本学习数据集"""
        random.seed(seed)
        
        # 按类别分组
        items_by_class = defaultdict(list)
        for item in train_items:
            items_by_class[item.classname].append(item)
        
        fewshot_items = []
        for classname, class_items in items_by_class.items():
            selected = random.sample(class_items, min(num_shots, len(class_items)))
            fewshot_items.extend(selected)
        
        print(f"Generated few-shot dataset with {len(fewshot_items)} items ({num_shots} shots per class)")
        return fewshot_items

    def create_predefined_split_file(self, train_ratio=0.8, output_file=None):
        """
        创建预定义划分文件的辅助方法
        
        Args:
            train_ratio: 训练集比例
            output_file: 输出文件路径，如果为None则使用默认路径
        """
        if output_file is None:
            output_file = os.path.join(self.split_dir, "predefined_split.json")
        
        all_items = self._read_data_from_json(self.annotation_file)
        
        # 按类别分层划分
        items_by_class = defaultdict(list)
        for item in all_items:
            rel_path = os.path.relpath(item.impath, self.dataset_dir)
            items_by_class[item.classname].append(rel_path)
        
        train_files = []
        test_files = []
        
        for classname, file_paths in items_by_class.items():
            random.shuffle(file_paths)
            split_idx = int(len(file_paths) * train_ratio)
            
            train_files.extend(file_paths[:split_idx])
            test_files.extend(file_paths[split_idx:])
        
        split_data = {
            "train": train_files,
            "test": test_files,
            "metadata": {
                "total_files": len(train_files) + len(test_files),
                "train_count": len(train_files),
                "test_count": len(test_files),
                "train_ratio": train_ratio,
                "classes": list(items_by_class.keys())
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(split_data, f, indent=2)
        
        print(f"Created predefined split file: {output_file}")
        return output_file
