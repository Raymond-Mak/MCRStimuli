#!/usr/bin/env python3
"""
跨数据集批量测试脚本
测试Emoset模型在Emotion6数据集上的性能（多个caption版本）
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
        'file': 'cache/Emotion6_Emotion6_gpt_H_narracap_extended_Emotion6_val.jsonl',
        'name': 'High-level (H)',
        'short_name': 'H'
    },
    {
        'file': 'cache/Emotion6_Emotion6_gpt_HO_narracap_extended_Emotion6_val.jsonl',
        'name': 'High-level + Other (HO)',
        'short_name': 'HO'
    },
    {
        'file': 'cache/Emotion6_Emotion6_gpt_HS_narracap_extended_Emotion6_val.jsonl',
        'name': 'High-level + Scene (HS)',
        'short_name': 'HS'
    },
    {
        'file': 'cache/Emotion6_Emotion6_gpt_narracap_extended_Emotion6_cleaned_val.jsonl',
        'name': 'Cleaned',
        'short_name': 'cleaned'
    },
    {
        'file': 'cache/Emotion6_Emotion6_gpt_narracap_extended_Emotion6_emotion_only_val.jsonl',
        'name': 'Emotion Only',
        'short_name': 'emotion_only'
    },
    {
        'file': 'cache/Emotion6_Emotion6_gpt_narracap_extended_Emotion6_highlevel_feature_only_val.jsonl',
        'name': 'High-level Feature Only',
        'short_name': 'highlevel_feat'
    },
    {
        'file': 'cache/Emotion6_Emotion6_gpt_narracap_extended_Emotion6_lowlevel_feature_only_val.jsonl',
        'name': 'Low-level Feature Only',
        'short_name': 'lowlevel_feat'
    },
    {
        'file': 'cache/Emotion6_Emotion6_gpt_narracap_extended_Emotion6_midhighfeature_mixed_val.jsonl',
        'name': 'Mid-High Feature Mixed',
        'short_name': 'midhigh_mixed'
    },
    {
        'file': 'cache/Emotion6_Emotion6_gpt_narracap_extended_Emotion6_midlevel_feature_only_val.jsonl',
        'name': 'Mid-level Feature Only',
        'short_name': 'midlevel_feat'
    },
    {
        'file': 'cache/Emotion6_Emotion6_gpt_narracap_extended_Emotion6_parts_val.jsonl',
        'name': 'Parts',
        'short_name': 'parts'
    },
    {
        'file': 'cache/Emotion6_Emotion6_gpt_narracap_extended_Emotion6_reasoning_only_val.jsonl',
        'name': 'Reasoning Only',
        'short_name': 'reasoning_only'
    },
    {
        'file': 'cache/Emotion6_Emotion6_gpt_narracap_extended_Emotion6_val.jsonl',
        'name': 'Full/Original',
        'short_name': 'full'
    },
    {
        'file': 'cache/Emotion6_Emotion6_gpt_narracap_extended_Emotion6_without_last_sentence_val.jsonl',
        'name': 'Without Last Sentence',
        'short_name': 'no_last_sent'
    },
    {
        'file': 'cache/Emotion6_Emotion6_gpt_O_narracap_extended_Emotion6_val.jsonl',
        'name': 'Other Only (O)',
        'short_name': 'O'
    },
    {
        'file': 'cache/Emotion6_Emotion6_gpt_OS_narracap_extended_Emotion6_val.jsonl',
        'name': 'Other + Scene (OS)',
        'short_name': 'OS'
    },
    {
        'file': 'cache/Emotion6_Emotion6_gpt_S_narracap_extended_Emotion6_val.jsonl',
        'name': 'Scene Only (S)',
        'short_name': 'S'
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

    # Emoset(8类) -> Emotion6(6类) 的类别映射
    emoset_to_emotion6 = {
        0: 3,  # amusement -> happiness/enjoyment (joy)
        1: 0,  # anger -> anger
        2: 5,  # awe -> surprise
        3: 3,  # contentment -> happiness/enjoyment (joy)
        4: 1,  # disgust -> disgust
        5: 3,  # excitement -> happiness/enjoyment (joy)
        6: 2,  # fear -> fear
        7: 4,  # sadness -> sadness
    }

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

    # 3. 运行测试
    print("\n[3/3] 开始测试...")
    model.eval()
    all_preds_mapped = []
    all_labels_filtered = []
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

            # 应用类别映射
            preds_cpu = preds.cpu()
            labels_cpu = labels.cpu()
            preds_mapped = torch.tensor([emoset_to_emotion6[p.item()] for p in preds_cpu])

            preds_filtered = preds_mapped
            labels_filtered = labels_cpu

            # 统计
            correct += (preds_filtered == labels_filtered).sum().item()
            total += labels_filtered.size(0)

            all_preds_mapped.extend(preds_filtered.numpy())
            all_labels_filtered.extend(labels_filtered.numpy())

    # 计算结果
    accuracy = correct / total * 100

    print(f"\n✅ 测试完成")
    print(f"   测试样本数: {total}")
    print(f"   准确率: {accuracy:.4f}%")
    print(f"   正确预测数: {correct}/{total}")

    # 生成分类报告
    from sklearn.metrics import classification_report, confusion_matrix
    emotion_names = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]

    unique_labels = sorted(set(all_labels_filtered + all_preds_mapped))
    target_names = [emotion_names[i] for i in unique_labels if i < len(emotion_names)]

    report = classification_report(
        all_labels_filtered, all_preds_mapped,
        labels=unique_labels,
        target_names=target_names,
        digits=4
    )

    cm = confusion_matrix(all_labels_filtered, all_preds_mapped, labels=unique_labels)

    return {
        'name': variant_name,
        'short_name': variant_info['short_name'],
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'report': report,
        'confusion_matrix': cm,
        'target_names': target_names
    }


def save_results(all_results, output_dir):
    """保存所有测试结果"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. 保存每个变体的详细结果
    for result in all_results:
        if result is None:
            continue

        filename = f"emotion6_{result['short_name']}_results.txt"
        filepath = os.path.join(output_dir, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"Emoset模型在Emotion6上的测试结果\n")
            f.write(f"Caption变体: {result['name']}\n")
            f.write(f"准确率: {result['accuracy']:.4f}%\n")
            f.write(f"正确预测数: {result['correct']}/{result['total']}\n\n")
            f.write("="*80 + "\n")
            f.write("分类报告:\n")
            f.write("="*80 + "\n")
            f.write(result['report'])
            f.write("\n\n" + "="*80 + "\n")
            f.write("混淆矩阵:\n")
            f.write("="*80 + "\n")
            f.write(str(result['confusion_matrix']))
            f.write(f"\n\n标签顺序: {result['target_names']}\n")

        print(f"   已保存: {filepath}")

    # 2. 保存汇总报告
    summary_file = os.path.join(output_dir, f"emotion6_summary_{timestamp}.txt")

    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("Emoset模型在Emotion6上的批量测试结果汇总\n")
        f.write(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"测试变体数量: {len([r for r in all_results if r is not None])}\n")
        f.write("="*80 + "\n\n")

        # 按准确率排序
        valid_results = [r for r in all_results if r is not None]
        sorted_results = sorted(valid_results, key=lambda x: x['accuracy'], reverse=True)

        f.write("排名 | Caption变体 | 准确率 | 正确数/总数\n")
        f.write("-"*80 + "\n")

        for i, result in enumerate(sorted_results, 1):
            f.write(f"{i:2d}   | {result['name']:30s} | {result['accuracy']:6.2f}% | {result['correct']:4d}/{result['total']:4d}\n")

        f.write("\n" + "="*80 + "\n")
        f.write("详细说明:\n")
        f.write("="*80 + "\n")
        f.write("- H: High-level features only\n")
        f.write("- HO: High-level + Other features\n")
        f.write("- HS: High-level + Scene features\n")
        f.write("- O: Other features only\n")
        f.write("- OS: Other + Scene features\n")
        f.write("- S: Scene features only\n")
        f.write("- cleaned: Cleaned version of full captions\n")
        f.write("- emotion_only: Emotion words only\n")
        f.write("- highlevel_feat: High-level feature descriptions\n")
        f.write("- lowlevel_feat: Low-level feature descriptions\n")
        f.write("- midhigh_mixed: Mid-level and high-level mixed\n")
        f.write("- midlevel_feat: Mid-level feature descriptions\n")
        f.write("- parts: Object parts description\n")
        f.write("- reasoning_only: Reasoning/explanation only\n")
        f.write("- full: Full/Original captions\n")
        f.write("- no_last_sent: Without last sentence\n")

    print(f"\n✅ 汇总报告已保存: {summary_file}")

    # 3. 打印汇总到控制台
    print("\n" + "="*80)
    print("测试结果汇总（按准确率排序）")
    print("="*80)
    print(f"{'排名':<4} {'Caption变体':<35} {'准确率':<10} {'正确数/总数'}")
    print("-"*80)

    for i, result in enumerate(sorted_results, 1):
        print(f"{i:<4} {result['name']:<35} {result['accuracy']:>6.2f}%     {result['correct']}/{result['total']}")


def main():
    print("="*80)
    print("Emoset模型 -> Emotion6数据集批量测试")
    print("="*80)
    print(f"总共测试 {len(CAPTION_VARIANTS)} 个caption变体")
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
