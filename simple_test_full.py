#!/usr/bin/env python3
"""
跨数据集批量测试脚本：Emoset模型 -> FI数据集（多个caption版本）

类别映射说明：
  Emoset(小写): ["amusement", "anger", "awe", "contentment", "disgust", "excitement", "fear", "sadness"]
  FI(大写):    ["Amusement",  "Anger", "Awe", "Contentment",  "Disgust",  "Excitement",  "Fear",  "Sadness"]

标签索引已对齐：0->0, 1->1, ..., 7->7
无需额外映射，直接使用标签索引即可
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import numpy as np
from tqdm import tqdm
import os
from datetime import datetime

# 复用现有组件
from src.mm_model import MultiModalModel
from utils.pairs_dataset import ImageTextPairs
from utils.model_utils import te_transform


# 定义所有要测试的caption变体
CAPTION_VARIANTS = [
    {
        'file': 'cache/FI_new_FI_new_gpt_narracap_extended_FI_new_val.jsonl',
        'name': 'FI New Full',
        'short_name': 'FI_new_full'
    },
    {
        'file': 'cache/FI_Probing_FI_Probing_gpt_narracap_extended_FI_Probing_cleaned_val.jsonl',
        'name': 'Cleaned',
        'short_name': 'cleaned'
    },
    {
        'file': 'cache/FI_Probing_FI_Probing_gpt_narracap_extended_FI_Probing_emotion_only_val.jsonl',
        'name': 'Emotion Only',
        'short_name': 'emotion_only'
    },
    {
        'file': 'cache/FI_Probing_FI_Probing_gpt_narracap_extended_FI_Probing_highlevel_feature_only_val.jsonl',
        'name': 'High-level Feature Only',
        'short_name': 'highlevel_feat'
    },
    {
        'file': 'cache/FI_Probing_FI_Probing_gpt_narracap_extended_FI_Probing_lowlevel_feature_only_val.jsonl',
        'name': 'Low-level Feature Only',
        'short_name': 'lowlevel_feat'
    },
    {
        'file': 'cache/FI_Probing_FI_Probing_gpt_narracap_extended_FI_Probing_midhighfeature_mixed_val.jsonl',
        'name': 'Mid-High Feature Mixed',
        'short_name': 'midhigh_mixed'
    },
    {
        'file': 'cache/FI_Probing_FI_Probing_gpt_narracap_extended_FI_Probing_midlevel_feature_only_val.jsonl',
        'name': 'Mid-level Feature Only',
        'short_name': 'midlevel_feat'
    },
    {
        'file': 'cache/FI_Probing_FI_Probing_gpt_narracap_extended_FI_Probing_parts_val.jsonl',
        'name': 'Parts',
        'short_name': 'parts'
    },
    {
        'file': 'cache/FI_Probing_FI_Probing_gpt_narracap_extended_FI_Probing_reasoning_only_val.jsonl',
        'name': 'Reasoning Only',
        'short_name': 'reasoning_only'
    },
    {
        'file': 'cache/FI_Probing_FI_Probing_gpt_narracap_extended_FI_Probing_val.jsonl',
        'name': 'Full/Original (GPT)',
        'short_name': 'full_gpt'
    },
    {
        'file': 'cache/FI_Probing_FI_Probing_gpt_narracap_extended_FI_Probing_without_last_sentence_val.jsonl',
        'name': 'Without Last Sentence',
        'short_name': 'no_last_sent'
    },
    {
        'file': 'cache/FI_Probing_FI_Probing_narracap_extended_FI_Probing_val.jsonl',
        'name': 'Original (No GPT)',
        'short_name': 'original'
    },
]


def load_model():
    """加载模型和tokenizer"""
    print("=== 加载Emoset模型 ===")

    # 1. 加载checkpoint
    print("\n[1/3] 加载checkpoint...")
    ckpt = torch.load('checkpoints/mm_Emoset/best_model.pt', map_location='cpu')
    state_dict = ckpt['model_state_dict']

    # 从state_dict推断num_classes
    num_classes = state_dict['fusion_head.classifier.weight'].shape[0]
    print(f"   检测到 {num_classes} 个类别")

    # 2. 创建模型并加载权重
    print("\n[2/3] 创建模型...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   使用设备: {device}")

    model = MultiModalModel(
        num_classes=num_classes,
        encoder_type='clip',
        vision_model_name='ViT-B/32',
        bert_model_name='roberta-large',
        freeze_bert=True,
        fusion_type='standard'
    )
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    print("   模型加载完成")

    # 3. 加载tokenizer
    print("\n[3/3] 加载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('roberta-large')
    print("   Tokenizer加载完成")

    return model, tokenizer, device


def test_variant(model, tokenizer, device, variant_info):
    """测试单个caption变体"""
    variant_name = variant_info['name']
    variant_file = variant_info['file']

    print(f"\n{'='*80}")
    print(f"正在测试: {variant_name}")
    print(f"文件: {variant_file}")
    print(f"{'='*80}")

    # FI的8个类别（与Emoset一一对应）
    EMOTION_NAMES = ["Amusement", "Anger", "Awe", "Contentment",
                     "Disgust", "Excitement", "Fear", "Sadness"]

    # 检查文件是否存在
    if not os.path.exists(variant_file):
        print(f"⚠️  文件不存在，跳过: {variant_file}")
        return None

    # 1. 创建测试数据集
    print("\n[1/3] 创建测试数据集...")
    try:
        val_dataset = ImageTextPairs(
            pairs_file=variant_file,
            img_root='data',
            tokenizer=tokenizer,
            transform=te_transform
        )
        print(f"   测试集大小: {len(val_dataset)}")
    except Exception as e:
        print(f"❌ 创建数据集失败: {e}")
        return None

    # 2. 创建DataLoader
    print("\n[2/3] 创建DataLoader...")
    test_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # 3. 运行测试（不过滤任何类别）
    print("\n[3/3] 开始测试...")
    model.eval()
    all_preds = []
    all_labels = []
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"测试 {variant_info['short_name']}", leave=False):
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            # 前向传播
            logits = model(images, input_ids, attention_mask)
            preds = logits.argmax(dim=1)

            # 直接计算，不过滤
            preds_cpu = preds.cpu()
            labels_cpu = labels.cpu()

            # 统计
            correct += (preds_cpu == labels_cpu).sum().item()
            total += labels_cpu.size(0)

            all_preds.extend(preds_cpu.numpy())
            all_labels.extend(labels_cpu.numpy())

    # 计算结果
    accuracy = correct / total * 100

    print(f"\n✅ 测试完成")
    print(f"   测试样本数: {total} (全部8个类别)")
    print(f"   准确率: {accuracy:.4f}%")
    print(f"   正确预测数: {correct}/{total}")

    # 生成分类报告
    from sklearn.metrics import classification_report, confusion_matrix

    report = classification_report(
        all_labels, all_preds,
        labels=range(len(EMOTION_NAMES)),
        target_names=EMOTION_NAMES,
        digits=4
    )

    cm = confusion_matrix(all_labels, all_preds, labels=range(len(EMOTION_NAMES)))

    # 计算每个类别的准确率
    class_accuracies = {}
    for i, name in enumerate(EMOTION_NAMES):
        mask = np.array(all_labels) == i
        if mask.sum() > 0:
            class_preds = np.array(all_preds)[mask]
            class_labels = np.array(all_labels)[mask]
            class_acc = (class_preds == class_labels).sum() / mask.sum() * 100
            class_accuracies[name] = {
                'accuracy': class_acc,
                'correct': (class_preds == class_labels).sum(),
                'total': mask.sum()
            }

    return {
        'name': variant_name,
        'short_name': variant_info['short_name'],
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'report': report,
        'confusion_matrix': cm,
        'emotion_names': EMOTION_NAMES,
        'class_accuracies': class_accuracies
    }


def save_results(all_results, output_dir):
    """保存所有测试结果"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. 保存每个变体的详细结果
    for result in all_results:
        if result is None:
            continue

        filename = f"fi_{result['short_name']}_results.txt"
        filepath = os.path.join(output_dir, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"Emoset模型在FI数据集上的测试结果\n")
            f.write(f"Caption变体: {result['name']}\n")
            f.write(f"准确率: {result['accuracy']:.4f}%\n")
            f.write(f"正确预测数: {result['correct']}/{result['total']}\n\n")
            f.write("="*80 + "\n")
            f.write("每个类别的准确率:\n")
            f.write("="*80 + "\n")
            for emotion, metrics in result['class_accuracies'].items():
                f.write(f"  {emotion:12s}: {metrics['accuracy']:6.2f}% "
                       f"({metrics['correct']}/{metrics['total']})\n")
            f.write("\n" + "="*80 + "\n")
            f.write("分类报告:\n")
            f.write("="*80 + "\n")
            f.write(result['report'])
            f.write("\n\n" + "="*80 + "\n")
            f.write("混淆矩阵:\n")
            f.write("="*80 + "\n")
            f.write(str(result['confusion_matrix']))
            f.write(f"\n\n标签顺序: {result['emotion_names']}\n")

        print(f"   已保存: {filepath}")

    # 2. 保存汇总报告
    summary_file = os.path.join(output_dir, f"fi_summary_{timestamp}.txt")

    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("Emoset模型在FI数据集上的批量测试结果汇总\n")
        f.write(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"测试变体数量: {len([r for r in all_results if r is not None])}\n")
        f.write("="*80 + "\n\n")

        # 按准确率排序
        valid_results = [r for r in all_results if r is not None]
        sorted_results = sorted(valid_results, key=lambda x: x['accuracy'], reverse=True)

        f.write("排名 | Caption变体 | 准确率 | 正确数/总数\n")
        f.write("-"*80 + "\n")

        for i, result in enumerate(sorted_results, 1):
            f.write(f"{i:2d}   | {result['name']:30s} | {result['accuracy']:6.2f}% | "
                   f"{result['correct']:4d}/{result['total']:4d}\n")

        f.write("\n" + "="*80 + "\n")
        f.write("各变体在8个FI类别上的详细准确率:\n")
        f.write("="*80 + "\n\n")

        # 为每个类别创建排名表
        for emotion in ["Amusement", "Anger", "Awe", "Contentment",
                       "Disgust", "Excitement", "Fear", "Sadness"]:
            f.write(f"\n{emotion}:\n")
            f.write("-"*80 + "\n")

            # 按该类别准确率排序
            emotion_results = []
            for result in sorted_results:
                if emotion in result['class_accuracies']:
                    emotion_results.append({
                        'name': result['name'],
                        'accuracy': result['class_accuracies'][emotion]['accuracy'],
                        'correct': result['class_accuracies'][emotion]['correct'],
                        'total': result['class_accuracies'][emotion]['total']
                    })

            emotion_sorted = sorted(emotion_results, key=lambda x: x['accuracy'], reverse=True)

            for i, er in enumerate(emotion_sorted, 1):
                f.write(f"  {i:2d}. {er['name']:30s} | {er['accuracy']:6.2f}% | "
                       f"{er['correct']:3d}/{er['total']:3d}\n")

        f.write("\n" + "="*80 + "\n")
        f.write("详细说明:\n")
        f.write("="*80 + "\n")
        f.write("- FI_new_full: New FI dataset with full GPT captions\n")
        f.write("- cleaned: Cleaned version of full captions\n")
        f.write("- emotion_only: Emotion words only\n")
        f.write("- highlevel_feat: High-level feature descriptions\n")
        f.write("- lowlevel_feat: Low-level feature descriptions\n")
        f.write("- midhigh_mixed: Mid-level and high-level mixed\n")
        f.write("- midlevel_feat: Mid-level feature descriptions\n")
        f.write("- parts: Object parts description\n")
        f.write("- reasoning_only: Reasoning/explanation only\n")
        f.write("- full_gpt: Full/Original captions with GPT\n")
        f.write("- no_last_sent: Without last sentence\n")
        f.write("- original: Original captions without GPT processing\n")

    print(f"\n✅ 汇总报告已保存: {summary_file}")

    # 3. 打印汇总到控制台
    print("\n" + "="*80)
    print("测试结果汇总（按准确率排序）")
    print("="*80)
    print(f"{'排名':<4} {'Caption变体':<35} {'准确率':<10} {'正确数/总数'}")
    print("-"*80)

    for i, result in enumerate(sorted_results, 1):
        print(f"{i:<4} {result['name']:<35} {result['accuracy']:>6.2f}%     "
              f"{result['correct']}/{result['total']}")

    # 显示每个类别的最佳表现
    print("\n" + "="*80)
    print("每个类别上表现最好的caption变体")
    print("="*80)

    for emotion in ["Amusement", "Anger", "Awe", "Contentment",
                   "Disgust", "Excitement", "Fear", "Sadness"]:
        best_result = None
        best_acc = 0
        for result in sorted_results:
            if emotion in result['class_accuracies']:
                acc = result['class_accuracies'][emotion]['accuracy']
                if acc > best_acc:
                    best_acc = acc
                    best_result = result

        if best_result:
            metrics = best_result['class_accuracies'][emotion]
            print(f"{emotion:12s}: {best_result['name']:35s} | "
                  f"{metrics['accuracy']:6.2f}% | {metrics['correct']}/{metrics['total']}")


def main():
    print("="*80)
    print("Emoset模型 -> FI数据集批量测试")
    print("="*80)
    print(f"总共测试 {len(CAPTION_VARIANTS)} 个caption变体")
    print(f"FI数据集有8个类别，与Emoset一一对应")
    print("="*80)

    # 创建输出目录
    output_dir = 'test_results'
    os.makedirs(output_dir, exist_ok=True)

    # 加载模型（只加载一次）
    model, tokenizer, device = load_model()

    # 测试所有变体
    all_results = []

    for i, variant in enumerate(CAPTION_VARIANTS, 1):
        print(f"\n进度: [{i}/{len(CAPTION_VARIANTS)}]")
        result = test_variant(model, tokenizer, device, variant)
        all_results.append(result)

    # 保存结果
    print("\n" + "="*80)
    print("保存测试结果...")
    print("="*80)
    save_results(all_results, output_dir)

    print("\n" + "="*80)
    print("✅ 所有测试完成！")
    print("="*80)


if __name__ == '__main__':
    main()
