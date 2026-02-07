import os
import random
import pickle
from collections import defaultdict
from dassl.data.datasets import DATASET_REGISTRY, DatasetBase
from dassl.utils import listdir_nohidden, mkdir_if_missing

# 数据集的8个情绪类别
FI_Probing_EMOTIONS = ["amusement", "anger", "awe", "contentment", "disgust", "excitement", "fear", "sadness"]
# 建立情绪名称到整数标签的映射
CLASSNAME_TO_LABEL = {name: i for i, name in enumerate(FI_Probing_EMOTIONS)}
LABEL_TO_CLASSNAME = {i: name for i, name in enumerate(FI_Probing_EMOTIONS)}
NUM_CLASSES = len(FI_Probing_EMOTIONS)


class Datum:
    """数据项结构"""
    def __init__(self, impath, label, domain=0, classname=None):
        self.impath = impath
        self.label = label
        self.domain = domain
        self.classname = classname


@DATASET_REGISTRY.register()
class FI_Probing(DatasetBase):
    dataset_dir = "FI_Probing"
    domains = []

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)

        # 创建用于保存few-shot数据的目录
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        # 验证数据集结构
        self._validate_dataset_structure()

        # 从文件夹中读取数据
        print("Loading train data...")
        train_items = self._read_data_from_folder("train")
        print("Loading test data...")
        test_items = self._read_data_from_folder("test")


        # 处理类别子采样
        subsample = getattr(cfg.DATASET, 'SUBSAMPLE_CLASSES', 'all')
        if subsample != 'all':
            train_items, test_items = self._subsample_classes(
                train_items, test_items, subsample=subsample
            )

        super().__init__(train_x=train_items,  test=test_items)

    def _validate_dataset_structure(self):
        """验证数据集结构是否正确"""
        if not os.path.exists(self.dataset_dir):
            raise FileNotFoundError(f"Dataset directory not found: {self.dataset_dir}")

        # 检查必需的分裂文件夹
        required_splits = ["train", "test"]
        missing_splits = [split for split in required_splits
                         if not os.path.exists(os.path.join(self.dataset_dir, split))]

        if missing_splits:
            raise FileNotFoundError(f"Missing split directories: {missing_splits}")

        print(f"Dataset validation passed. Found all required split directories.")

    def _validate_file_path(self, file_path, allowed_directory):
        """验证文件路径，防止路径遍历攻击"""
        try:
            resolved_path = os.path.realpath(file_path)
            allowed_dir = os.path.realpath(allowed_directory)
            return resolved_path.startswith(allowed_dir)
        except Exception:
            return False

    def _read_data_from_folder(self, split_name):
        """从指定分裂的文件夹中读取数据"""
        split_dir = os.path.join(self.dataset_dir, split_name)
        all_items = []

        print(f"Reading {split_name} data from {split_dir}")

        # 获取split目录下的所有情绪文件夹
        try:
            emotion_folders = [f for f in os.listdir(split_dir)
                              if os.path.isdir(os.path.join(split_dir, f))]
        except Exception as e:
            raise RuntimeError(f"Failed to read directory {split_dir}: {e}")

        for emotion_folder in emotion_folders:
            emotion = emotion_folder.lower()  # 简单转换为小写
            emotion_dir = os.path.join(split_dir, emotion_folder)

            # 检查是否是有效的情绪类别
            if emotion not in FI_Probing_EMOTIONS:
                print(f"Warning: Unknown emotion folder '{emotion_folder}', skipping.")
                continue

            # 获取图像文件
            try:
                image_files = listdir_nohidden(emotion_dir)
                image_files = [f for f in image_files
                             if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
            except Exception as e:
                print(f"Warning: Error reading directory {emotion_dir}: {e}. Skipping.")
                continue

            print(f"Loading {len(image_files)} images from {split_name}/{emotion_folder}")

            for image_file in image_files:
                impath = os.path.join(emotion_dir, image_file)

                # 安全验证
                if not self._validate_file_path(impath, self.dataset_dir):
                    print(f"Warning: Invalid file path: {impath}. Skipping.")
                    continue

                if not os.path.exists(impath) or not os.path.isfile(impath):
                    print(f"Warning: File not found or invalid: {impath}. Skipping.")
                    continue

                label = CLASSNAME_TO_LABEL[emotion]
                item = Datum(impath=impath, label=label, domain=0, classname=emotion)
                all_items.append(item)

        if not all_items:
            raise ValueError(f"No valid data items were loaded from {split_name} split.")

        print(f"Successfully loaded {len(all_items)} items from {split_name} split.")
        return all_items

    def _generate_fewshot_dataset(self, items, num_shots, seed):
        """生成少样本学习数据集"""
        random.seed(seed)

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

    def _subsample_classes(self, train, test, subsample):
        """子采样类别"""
        if subsample == 'all':
            return train, test

        if isinstance(subsample, int):
            selected_classes = FI_Probing_EMOTIONS[:subsample]
        elif isinstance(subsample, list):
            selected_classes = [cls for cls in subsample if cls in FI_Probing_EMOTIONS]
        else:
            raise ValueError(f"Invalid subsample parameter: {subsample}")

        print(f"Subsampling to classes: {selected_classes}")

        def filter_by_classes(items, classes):
            return [item for item in items if item.classname in classes]

        train = filter_by_classes(train, selected_classes)
        test = filter_by_classes(test, selected_classes)

        return train, test