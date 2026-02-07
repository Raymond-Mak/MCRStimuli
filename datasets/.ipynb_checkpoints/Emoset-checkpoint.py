import os
import json
import random
import pickle
import numpy as np
import math
from collections import defaultdict
from dassl.data.datasets import DATASET_REGISTRY, DatasetBase
from dassl.utils import listdir_nohidden, mkdir_if_missing

# EmoSet数据集的8个情绪类别
EMOSET_EMOTIONS = ["amusement", "anger", "awe", "contentment", "disgust", "excitement", "fear", "sadness"]
# 建立情绪名称到整数标签的映射
CLASSNAME_TO_LABEL = {name: i for i, name in enumerate(EMOSET_EMOTIONS)}
LABEL_TO_CLASSNAME = {i: name for i, name in enumerate(EMOSET_EMOTIONS)}
NUM_CLASSES = len(EMOSET_EMOTIONS)


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
class Emoset(DatasetBase):
    dataset_dir = ""  # 数据集文件夹名称
    domains = []

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        
        # 设置各个子目录的路径
        self.image_dir = os.path.join(self.dataset_dir, "")
        self.annotation_dir = os.path.join(self.dataset_dir, "annotation")
        
        # JSON标注文件路径
        self.train_json = os.path.join(self.dataset_dir, "train.json")
        self.test_json = os.path.join(self.dataset_dir, "test.json")
        self.val_json = os.path.join(self.dataset_dir, "val.json")
        self.info_json = os.path.join(self.dataset_dir, "info.json")
        
        # 创建用于保存few-shot数据的目录
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)
        
          
        # 验证必要文件是否存在
        self._validate_dataset_structure()
        
        # 读取数据集划分
        train_items = self._read_data_from_json(self.train_json, "train")
        val_items = self._read_data_from_json(self.val_json, "validation")
        test_items = self._read_data_from_json(self.test_json, "test")
        
        # 处理少样本学习配置
        num_shots = cfg.DATASET.NUM_SHOTS
        if num_shots >= 1:
            seed = cfg.SEED
            preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")
            
            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train_items, val_items = data["train"], data["val"]
            else:
                train_items = self._generate_fewshot_dataset(train_items, num_shots=num_shots, seed=seed)
                val_items = self._generate_fewshot_dataset(val_items, num_shots=min(num_shots, 4), seed=seed)
                data = {"train": train_items, "val": val_items}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)
        
        # 处理类别子采样
        subsample = getattr(cfg.DATASET, 'SUBSAMPLE_CLASSES', 'all')
        if subsample != 'all':
            train_items, val_items, test_items = self._subsample_classes(
                train_items, val_items, test_items, subsample=subsample
            )
        
        super().__init__(train_x=train_items, val=val_items, test=test_items)

    def _validate_dataset_structure(self):
        """验证数据集结构是否正确"""
        required_files = [self.train_json, self.test_json, self.val_json]
        missing_files = [f for f in required_files if not os.path.exists(f)]
        
        if missing_files:
            raise FileNotFoundError(f"Missing required files: {missing_files}")
        
        if not os.path.exists(self.image_dir):
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        
        # 检查情绪类别文件夹是否存在
        missing_emotion_dirs = []
        for emotion in EMOSET_EMOTIONS:
            emotion_dir = os.path.join(self.image_dir, emotion)
            if not os.path.exists(emotion_dir):
                missing_emotion_dirs.append(emotion)
        
        if missing_emotion_dirs:
            print(f"Warning: Missing emotion directories: {missing_emotion_dirs}")

    def _read_data_from_json(self, json_file, split_name):
        """从JSON文件读取数据项"""
        if not os.path.exists(json_file):
            raise FileNotFoundError(f"JSON file not found: {json_file}")
        
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        items = []
        print(f"Reading {split_name} data from {json_file}")
        
        # 检查JSON格式并相应处理
        if isinstance(data, list):
            # 检查列表中的元素格式
            if len(data) > 0:
                first_item = data[0]
                if isinstance(first_item, list) and len(first_item) >= 3:
                    # EmoSet格式: [emotion, image_path, annotation_path]
                    return self._read_emoset_array_format(data, split_name)
                elif isinstance(first_item, dict):
                    # 字典列表格式
                    return self._read_dict_list_format(data, split_name)
                else:
                    # 其他列表格式
                    return self._read_other_list_format(data, split_name)
        elif isinstance(data, dict):
            # 字典格式
            return self._read_dict_format(data, split_name)
        else:
            raise ValueError(f"Unsupported JSON format in {json_file}")
    
    def _read_emoset_array_format(self, data, split_name):
        """读取EmoSet数组格式: [emotion, image_path, annotation_path]"""
        items = []
        
        for i, item in enumerate(data):
            if not isinstance(item, list) or len(item) < 2:
                print(f"Warning: Invalid item format at index {i}. Expected list with at least 2 elements. Skipping.")
                continue
            
            emotion = item[0]
            image_path = item[1]
            # annotation_path = item[2] if len(item) > 2 else None
            
            # 验证情绪类别
            if not emotion or emotion not in EMOSET_EMOTIONS:
                print(f"Warning: Unknown emotion '{emotion}' for image {image_path}. Skipping.")
                continue
            
            # 构建完整的图像路径
            full_image_path = self._build_image_path(image_path, emotion)
            if full_image_path is None:
                print(f"Warning: Image file not found for {image_path}. Skipping.")
                continue
            
            label = CLASSNAME_TO_LABEL[emotion]
            domain = 0
            
            item_obj = Datum(impath=full_image_path, label=label, domain=domain,
                           classname=emotion)
            items.append(item_obj)
        
        if not items:
            raise ValueError(f"No valid data items were loaded from {split_name} split.")
        
        print(f"Successfully loaded {len(items)} items from {split_name} split (EmoSet array format).")
        return items
    
    def _read_dict_list_format(self, data, split_name):
        """读取字典列表格式"""
        items = []
        
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                print(f"Warning: Invalid item format at index {i}. Expected dictionary. Skipping.")
                continue
            
            emotion = item.get("emotion") or item.get("label") or item.get("true_emotion")
            image_path = item.get("image_path") or item.get("path") or item.get("image")
            
            if not emotion or emotion not in EMOSET_EMOTIONS:
                print(f"Warning: Unknown emotion '{emotion}' for item {i}. Skipping.")
                continue
            
            if not image_path:
                print(f"Warning: No image path found for item {i}. Skipping.")
                continue
            
            full_image_path = self._build_image_path(image_path, emotion)
            if full_image_path is None:
                print(f"Warning: Image file not found for {image_path}. Skipping.")
                continue
            
            label = CLASSNAME_TO_LABEL[emotion]
            domain = 0
            
            item_obj = Datum(impath=full_image_path, label=label, domain=domain,
                           classname=emotion)
            items.append(item_obj)
        
        if not items:
            raise ValueError(f"No valid data items were loaded from {split_name} split.")
        
        print(f"Successfully loaded {len(items)} items from {split_name} split (dict list format).")
        return items
    
    def _read_other_list_format(self, data, split_name):
        """读取其他列表格式"""
        items = []
        
        for i, item in enumerate(data):
            if isinstance(item, str):
                # 尝试从文件名推断情绪
                emotion = None
                for emo in EMOSET_EMOTIONS:
                    if emo in item.lower():
                        emotion = emo
                        break
                
                if not emotion:
                    print(f"Warning: Cannot infer emotion from filename '{item}'. Skipping.")
                    continue
                
                full_image_path = self._build_image_path(item, emotion)
                if full_image_path is None:
                    print(f"Warning: Image file not found for {item}. Skipping.")
                    continue
                
                label = CLASSNAME_TO_LABEL[emotion]
                domain = 0
                
                item_obj = Datum(impath=full_image_path, label=label, domain=domain,
                               classname=emotion)
                items.append(item_obj)
        
        if not items:
            raise ValueError(f"No valid data items were loaded from {split_name} split.")
        
        print(f"Successfully loaded {len(items)} items from {split_name} split (other list format).")
        return items
    
    def _read_dict_format(self, data, split_name):
        """读取字典格式"""
        items = []
        
        # 检查是否有嵌套的data字段
        if 'data' in data:
            data = data['data']
        
        for key, value in data.items():
            if isinstance(value, dict):
                emotion = value.get("emotion") or value.get("label") or value.get("true_emotion")
                image_path = value.get("image_path") or value.get("path") or str(key)
            else:
                # 尝试从key推断
                emotion = None
                image_path = str(key)
                for emo in EMOSET_EMOTIONS:
                    if emo in image_path.lower():
                        emotion = emo
                        break
            
            if not emotion or emotion not in EMOSET_EMOTIONS:
                print(f"Warning: Unknown emotion '{emotion}' for key '{key}'. Skipping.")
                continue
            
            full_image_path = self._build_image_path(image_path, emotion)
            if full_image_path is None:
                print(f"Warning: Image file not found for {image_path}. Skipping.")
                continue
            
            label = CLASSNAME_TO_LABEL[emotion]
            domain = 0
            
            item_obj = Datum(impath=full_image_path, label=label, domain=domain,
                           classname=emotion)
            items.append(item_obj)
        
        if not items:
            raise ValueError(f"No valid data items were loaded from {split_name} split.")
        
        print(f"Successfully loaded {len(items)} items from {split_name} split (dict format).")
        return items
    
    def _build_image_path(self, image_path, emotion):
        """构建并验证图像路径"""
        if os.path.isabs(image_path) and os.path.exists(image_path):
            return image_path
        
        # 尝试多种可能的路径组合
        possible_paths = [
            # 直接在数据集目录下
            os.path.join(self.dataset_dir, image_path),
            # 在image目录下
            os.path.join(self.image_dir, image_path),
            # 在情绪子目录下
            os.path.join(self.image_dir, emotion, os.path.basename(image_path)),
            # 去掉可能的前缀路径
            os.path.join(self.image_dir, os.path.basename(image_path)),
            # 如果路径已经包含emotion，直接使用
            os.path.join(self.dataset_dir, image_path) if emotion in image_path else None
        ]
        
        # 过滤掉None值
        possible_paths = [p for p in possible_paths if p is not None]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        # 如果都找不到，打印调试信息
        print(f"Debug: Tried paths for {image_path}:")
        for path in possible_paths[:3]:  # 只打印前3个避免太长
            print(f"  - {path} (exists: {os.path.exists(path)})")
        
        return None

    def _generate_fewshot_dataset(self, items, num_shots, seed):
        """生成少样本学习数据集"""
        random.seed(seed)
        
        # 按类别分组
        items_by_class = defaultdict(list)
        for item in items:
            items_by_class[item.classname].append(item)
        
        fewshot_items = []
        for classname, class_items in items_by_class.items():
            if len(class_items) >= num_shots:
                selected = random.sample(class_items, num_shots)
            else:
                selected = class_items
                print(f"Warning: Class '{classname}' has only {len(class_items)} samples, less than {num_shots} shots.")
            fewshot_items.extend(selected)
        
        print(f"Generated few-shot dataset with {len(fewshot_items)} items ({num_shots} shots per class)")
        return fewshot_items

    def _subsample_classes(self, train, val, test, subsample):
        """子采样类别"""
        if subsample == 'all':
            return train, val, test
        
        if isinstance(subsample, int):
            selected_classes = EMOSET_EMOTIONS[:subsample]
        elif isinstance(subsample, list):
            selected_classes = [cls for cls in subsample if cls in EMOSET_EMOTIONS]
        else:
            raise ValueError(f"Invalid subsample parameter: {subsample}")
        
        print(f"Subsampling to classes: {selected_classes}")
        
        def filter_by_classes(items, classes):
            return [item for item in items if item.classname in classes]
        
        train = filter_by_classes(train, selected_classes)
        val = filter_by_classes(val, selected_classes)
        test = filter_by_classes(test, selected_classes)
        
        return train, val, test