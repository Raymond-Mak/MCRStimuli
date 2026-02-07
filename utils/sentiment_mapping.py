"""
情感类别到二分类（neg/pos）的映射配置
支持 Emotion6, FI_new, Emoset 等数据集
"""

SENTIMENT_MAPPING = {
    "Emotion6": {
        "num_classes": 6,
        "class_names": ["anger", "disgust", "fear", "joy", "sadness", "surprise"],
        "label_to_binary": {
            0: 0,  # anger → neg
            1: 0,  # disgust → neg
            2: 0,  # fear → neg
            3: 1,  # joy → pos
            4: 0,  # sadness → neg
            5: 1,  # surprise → pos
        },
        "binary_names": ["neg", "pos"]
    },

    "FI_new": {
        "num_classes": 8,
        "class_names": ["amusement", "anger", "awe", "contentment", "disgust", "excitement", "fear", "sadness"],
        "label_to_binary": {
            0: 1,  # amusement → pos
            1: 0,  # anger → neg
            2: 1,  # awe → pos
            3: 1,  # contentment → pos
            4: 0,  # disgust → neg
            5: 1,  # excitement → pos
            6: 0,  # fear → neg
            7: 0,  # sadness → neg
        },
        "binary_names": ["neg", "pos"]
    },

    "FI_Probing": {
        "num_classes": 8,
        "class_names": ["amusement", "anger", "awe", "contentment", "disgust", "excitement", "fear", "sadness"],
        "label_to_binary": {
            0: 1,  # amusement → pos
            1: 0,  # anger → neg
            2: 1,  # awe → pos
            3: 1,  # contentment → pos
            4: 0,  # disgust → neg
            5: 1,  # excitement → pos
            6: 0,  # fear → neg
            7: 0,  # sadness → neg
        },
        "binary_names": ["neg", "pos"]
    },

    "Emoset": {
        "num_classes": 8,
        "class_names": ["amusement", "anger", "awe", "contentment", "disgust", "excitement", "fear", "sadness"],
        "label_to_binary": {
            0: 1,  # amusement → pos
            1: 0,  # anger → neg
            2: 1,  # awe → pos
            3: 1,  # contentment → pos
            4: 0,  # disgust → neg
            5: 1,  # excitement → pos
            6: 0,  # fear → neg
            7: 0,  # sadness → neg
        },
        "binary_names": ["neg", "pos"]
    },

    # Twitter 数据集（本身就是二分类）
    "Twitter1": {
        "num_classes": 2,
        "class_names": ["neg", "pos"],
        "label_to_binary": {
            0: 0,  # neg → neg
            1: 1,  # pos → pos
        },
        "binary_names": ["neg", "pos"]
    },

    "Twitter2": {
        "num_classes": 2,
        "class_names": ["neg", "pos"],
        "label_to_binary": {
            0: 0,  # neg → neg
            1: 1,  # pos → pos
        },
        "binary_names": ["neg", "pos"]
    }
}


def get_binary_mapping(dataset_name: str, label: int) -> int:
    """
    根据数据集名称和原始标签，返回二分类标签

    Args:
        dataset_name: 数据集名称（如 "Emotion6", "FI_new", "Emoset"）
        label: 原始类别标签

    Returns:
        二分类标签（0=neg, 1=pos）
    """
    if dataset_name in SENTIMENT_MAPPING:
        mapping = SENTIMENT_MAPPING[dataset_name]["label_to_binary"]
        return mapping.get(label, 0)  # 默认返回 neg (0)

    # 如果数据集不在配置中，返回原始标签（不映射）
    return label


def get_binary_class_names() -> list:
    """返回二分类的类别名称"""
    return ["neg", "pos"]


def get_dataset_info(dataset_name: str) -> dict:
    """
    获取数据集的完整信息

    Args:
        dataset_name: 数据集名称

    Returns:
        包含 num_classes, class_names, label_to_binary, binary_names 的字典
    """
    return SENTIMENT_MAPPING.get(dataset_name, {})
