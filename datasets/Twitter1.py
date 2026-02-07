import os
import random
import pickle
import numpy as np
from collections import defaultdict
from dassl.data.datasets import DATASET_REGISTRY, DatasetBase
from dassl.utils import listdir_nohidden, mkdir_if_missing

# Twitter1数据集的2个情绪类别
TWITTER1_EMOTIONS = ["neg", "pos"]
# 建立情绪名称到整数标签的映射
CLASSNAME_TO_LABEL = {name: i for i, name in enumerate(TWITTER1_EMOTIONS)}
LABEL_TO_CLASSNAME = {i: name for i, name in enumerate(TWITTER1_EMOTIONS)}
NUM_CLASSES = len(TWITTER1_EMOTIONS)


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
class Twitter1(DatasetBase):
    dataset_dir = ""  # 数据集文件夹名称
    domains = []

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = root

        # 创建用于保存few-shot数据的目录
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        # 获取随机种子
        self.split_seed = getattr(cfg.DATASET, 'SPLIT_SEED', cfg.SEED)
        print(f"Split seed: {self.split_seed}")

        # 验证必要目录是否存在
        self._validate_dataset_structure()

        # 读取所有数据
        all_items = self._read_data_from_directories()

        # 进行8:2随机划分训练集和测试集
        train_items, test_items = self._random_split(all_items, train_ratio=0.8, seed=self.split_seed)

        # 处理少样本学习配置
        num_shots = cfg.DATASET.NUM_SHOTS
        if num_shots >= 1:
            seed = cfg.SEED
            preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")

            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train_items = data["train"]
            else:
                train_items = self._generate_fewshot_dataset(train_items, num_shots=num_shots, seed=seed)
                data = {"train": train_items}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        # 处理类别子采样
        subsample = getattr(cfg.DATASET, 'SUBSAMPLE_CLASSES', 'all')
        if subsample != 'all':
            train_items, test_items = self._subsample_classes(
                train_items, test_items, subsample=subsample
            )

        super().__init__(train_x=train_items, val=[], test=test_items)

    def _validate_dataset_structure(self):
        """验证数据集结构是否正确"""
        if not os.path.exists(self.dataset_dir):
            raise FileNotFoundError(f"Dataset directory not found: {self.dataset_dir}")

        # 检查情绪类别文件夹是否存在
        missing_emotion_dirs = []
        for emotion in TWITTER1_EMOTIONS:
            emotion_dir = os.path.join(self.dataset_dir, emotion)
            if not os.path.exists(emotion_dir):
                missing_emotion_dirs.append(emotion)

        if missing_emotion_dirs:
            raise FileNotFoundError(f"Missing emotion directories: {missing_emotion_dirs}")

        print(f"Dataset validation passed. Found all {len(TWITTER1_EMOTIONS)} emotion directories.")

    def _read_data_from_directories(self):
        """从各个情感目录中读取图像数据"""
        all_items = []

        for emotion in TWITTER1_EMOTIONS:
            emotion_dir = os.path.join(self.dataset_dir, emotion)

            # 获取该情感目录下的所有图像文件
            image_files = listdir_nohidden(emotion_dir)
            image_files = [f for f in image_files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]

            print(f"Loading {len(image_files)} images from {emotion} directory")

            for image_file in image_files:
                impath = os.path.join(emotion_dir, image_file)

                # 验证图像文件是否存在
                if not os.path.exists(impath):
                    print(f"Warning: Image file not found: {impath}. Skipping.")
                    continue

                label = CLASSNAME_TO_LABEL[emotion]
                domain = 0
                classname = emotion

                item = Datum(impath=impath, label=label, domain=domain,
                           classname=classname)
                all_items.append(item)

        if not all_items:
            raise ValueError("No valid data items were loaded from Twitter1 dataset.")

        print(f"Successfully loaded {len(all_items)} items from Twitter1 dataset.")
        return all_items

    def _random_split(self, all_items, train_ratio=0.8, seed=42):
        """随机划分数据集为训练集和测试集"""
        print(f"Performing random split with ratio {train_ratio}:{1-train_ratio} using seed {seed}")

        # 设置随机种子
        random.seed(seed)
        np.random.seed(seed)

        # 按类别分组进行分层划分
        items_by_class = defaultdict(list)
        for item in all_items:
            items_by_class[item.classname].append(item)

        train_items = []
        test_items = []

        for classname, class_items in items_by_class.items():
            # 随机打乱该类别的样本
            random.shuffle(class_items)

            # 计算训练集样本数量
            num_train = int(len(class_items) * train_ratio)

            # 划分训练集和测试集
            train_items.extend(class_items[:num_train])
            test_items.extend(class_items[num_train:])

            print(f"  {classname}: {num_train} train, {len(class_items)-num_train} test")

        # 重新随机化整体顺序
        random.shuffle(train_items)
        random.shuffle(test_items)

        print(f"Random split completed: {len(train_items)} train, {len(test_items)} test")
        return train_items, test_items

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

    def _subsample_classes(self, train, test, subsample):
        """子采样类别"""
        if subsample == 'all':
            return train, test

        if isinstance(subsample, int):
            selected_classes = TWITTER1_EMOTIONS[:subsample]
        elif isinstance(subsample, list):
            selected_classes = [cls for cls in subsample if cls in TWITTER1_EMOTIONS]
        else:
            raise ValueError(f"Invalid subsample parameter: {subsample}")

        print(f"Subsampling to classes: {selected_classes}")

        def filter_by_classes(items, classes):
            return [item for item in items if item.classname in classes]

        train = filter_by_classes(train, selected_classes)
        test = filter_by_classes(test, selected_classes)

        return train, test