import os

if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
    # Reduce CUDA allocator fragmentation when large tensors get split up
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

import argparse
import torch
import datetime
import time
from pathlib import Path
from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer
from utils.utils import *
# WandB监控
from wandb_monitor import create_monitor
# custom
import datasets.oxford_flowers
import datasets.fgvc_aircraft
import datasets.dtd
import datasets.eurosat
import datasets.food101
import datasets.sun397
import datasets.ucf101
import datasets.imagenet_r
import datasets.imagenet
import datasets.imagenet_s
import datasets.imagenet_a
import datasets.caltech101
import datasets.cifar
import datasets.Emotion6
import datasets.Emoset
import datasets.FI_new
import datasets.FI_Probing
import datasets.FI
import datasets.Twitter1
import datasets.Twitter2
import trainers.LaFTer_trainers as lafter_uft
from utils.utils import *
# 增强的训练设置模块 - 支持选择性visual encoder解冻
from utils.enhanced_training_setup import setup_enhanced_lafter_training_utils

from text_supervisor import TextSupervisor, enhance_batch_with_text


def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    if args.source_domains:
        cfg.DATASET.SOURCE_DOMAINS = args.source_domains

    if args.target_domains:
        cfg.DATASET.TARGET_DOMAINS = args.target_domains

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head


def extend_cfg(cfg):
    """
    Add new cong variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    from yacs.config import CfgNode as CN

    cfg.TRAINER.COOP = CN()
    cfg.TRAINER.COOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.COOP.CSC = False  # class-specific context
    cfg.TRAINER.COOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COOP.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.COOP.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'

    cfg.TRAINER.COCOOP = CN()
    cfg.TRAINER.COCOOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.COCOOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COCOOP.PREC = "fp16"  # fp16, fp32, amp

    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new

    cfg.txt_cls = args.txt_cls
    cfg.gpt_prompts = args.gpt_prompts


def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 3. From input arguments
    reset_cfg(cfg, args)

    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)

    return cfg


def _sanitize_wandb_config(args):
    """Convert argparse Namespace into wandb-friendly dict."""
    def _convert(val):
        if isinstance(val, (str, int, float, bool)) or val is None:
            return val
        if isinstance(val, (list, tuple)):
            return [_convert(v) for v in val]
        if isinstance(val, dict):
            return {k: _convert(v) for k, v in val.items()}
        return str(val)

    if args is None:
        return {}
    return {k: _convert(v) for k, v in vars(args).items()}


class lossmeter:
    """Compute and store the average and current value.

    Examples::
        >>> # 1. Initialize a meter to record loss
        >>> losses = AverageMeter()
        >>> # 2. Update meter after every mini-batch update
        >>> losses.update(loss_value, batch_size)
    """

    def __init__(self, ema=False):
        """
        Args:
            ema (bool, optional): apply exponential moving average.
        """
        self.ema = ema
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if isinstance(val, torch.Tensor):
            val = val.item()

        self.val = val
        self.sum += val * n
        self.count += n

        if self.ema:
            self.avg = self.avg * 0.9 + self.val * 0.1
        else:
            self.avg = self.sum / self.count


def test(args, teloader, model):
    model.eval()
    top1 = AverageMeter('Acc@1', ':6.2f')
    top1_pl = AverageMeter('Acc@1', ':6.2f')
    one_hot = []
    one_hot_pl = []

    for i, (inputs) in enumerate(tqdm(teloader)):
        img = inputs["img"]
        labels = inputs["label"]

        if args.zero_shot:
            with torch.no_grad():
                output_pseudo_label = model(inputs.cuda(), zero_shot=True)
                _, predicted_pl = output_pseudo_label.max(1)
                one_hot_pl.append(predicted_pl.eq(labels.cuda()).cpu())
                acc1_pl = one_hot_pl[-1].sum().item() / len(labels)
                top1_pl.update(acc1_pl, len(labels))

        else:
            with torch.no_grad():
                inputs, labels = img.cuda(), labels.cuda()
                outputs = model(inputs, clip_eval=True)
                _, predicted = outputs.max(1)
                one_hot.append(predicted.eq(labels).cpu())
                acc1 = one_hot[-1].sum().item() / len(labels)
                top1.update(acc1, len(labels))

    if not args.zero_shot:
        return top1.avg * 100, top1_pl.avg * 100
    else:
        return top1_pl.avg * 100


def train_txt_cls(args, model):
    optimizer, _, _ = setup_text_training_utils(args, model)
    criteria = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    for i in tqdm(range(args.txt_epochs)):
        loss = model.train_txt_clas(criteria)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.txt_cls_init()

def print_image_text_pairs(tr_loader, text_supervisor, model, max_samples=50):
    """
    在图像微调前打印图像-文本对以验证组合正确性

    Args:
        tr_loader: 训练数据加载器
        text_supervisor: 文本监督器实例
        model: 模型实例（用于设备信息）
        max_samples: 最大打印样本数量
    """
    print("\n" + "="*80)
    print(" 图像-文本对组合验证 - 打印前{}个样本".format(max_samples))
    print("="*80)

    matched_count = 0
    unmatched_count = 0
    total_processed = 0

    for i, batch in enumerate(tr_loader):
        if total_processed >= max_samples:
            break

        # 获取标签信息
        labels = batch["label"]
        batch_size = len(labels)

        # 直接从batch中获取图像路径
        image_paths = batch.get("impath", [])

        for j in range(batch_size):
            if total_processed >= max_samples:
                break

            # 获取图像路径
            if j < len(image_paths):
                img_path = image_paths[j]
            else:
                img_path = "Unknown"

            # 获取标签
            label_idx = labels[j].item() if hasattr(labels[j], 'item') else labels[j]

            # 从模型中获取类别名称
            try:
                classname = model.classes[label_idx]
            except:
                classname = f"Class_{label_idx}"

            # 获取对应的文本描述
            text = text_supervisor.get_text_for_image_path(img_path) if text_supervisor else None

            # 打印信息
            print(f"\n样本 {total_processed + 1:2d}:")
            print(f"   图像路径: {img_path}")
            print(f"    标签: {label_idx} ({classname})")
            print(f"   文本描述: {text if text else '❌ 未找到匹配文本'}")

            if text:
                matched_count += 1
            else:
                unmatched_count += 1

            total_processed += 1

    # 打印统计信息
    print("\n" + "="*80)
    print(" 统计信息:")
    print(f"   成功匹配: {matched_count} 个")
    print(f"   未匹配: {unmatched_count} 个")
    print(f"  匹配率: {matched_count/(matched_count+unmatched_count)*100:.1f}%" if (matched_count+unmatched_count) > 0 else "N/A")
    print(f"   总样本: {total_processed} 个")
    print("="*80)
    print()

def train_lafter(args, model, tr_loader, val_loader):

    # first train text classifier
    train_txt_cls(args, model)

    all_acc = list()

    # 使用增强的训练设置（支持选择性visual encoder解冻）
    if args.enable_visual_unfreezing:
        print("=== Using Enhanced Training Setup with Selective Visual Unfreezing ===")
        print(f"📋 Dropout Configuration:")
        print(f"   - Adapter dropout: {args.adapter_dropout}")
        print(f"   - Projector dropout: {args.projector_dropout}")
        print(f"   - Visual dropout: {args.visual_dropout}")
        print(f"   - Note: Dropout mechanisms are only active when --enable_visual_unfreezing is enabled")

        optimizer, scheduler, criteria, dropout_manager = setup_enhanced_lafter_training_utils(
            args, model,
            enable_visual_unfreezing=True,
            visual_unfreeze_strategy=args.visual_unfreeze_strategy,
            visual_unfreeze_blocks=args.visual_unfreeze_blocks,
            visual_lr=args.visual_lr,
            # Dropout parameters
            adapter_dropout=args.adapter_dropout,
            projector_dropout=args.projector_dropout,
            visual_dropout=args.visual_dropout
        )

        # 显示dropout manager信息
        if dropout_manager and dropout_manager.get('enabled', False):
            print(f"✅ Dropout mechanisms activated successfully")
        else:
            print(f"⚠️  Dropout mechanisms are disabled (all dropout probabilities set to 0)")
    else:
        print("=== Using Standard Training Setup ===")
        print("ℹ️  Note: To enable dropout mechanisms, use --enable_visual_unfreezing flag")
        optimizer, scheduler, criteria = setup_lafter_training_utils(args, model)
    batch_time = lossmeter()
    data_time = lossmeter()
    losses = lossmeter()  # 训练损失跟踪器
    correct_predictions = 0  # 初始化正确预测计数
    total_samples = 0  # 初始化总样本计数

    for epoch in range(args.epochs):
        print(f'Epoch: {epoch}')
        model.eval()
        model.adapter.train()
        end = time.time()

        # 重置每个epoch的统计信息
        losses.reset()
        correct_predictions = 0
        total_samples = 0

        for i, batch in enumerate((tr_loader)):
            data_time.update(time.time() - end)
            batch_time.update(time.time() - end)

            input = batch["img"]
            labels = batch["label"]  # 从batch中获取标签
            input = torch.stack(input)  # two views from dataloader
            input = input.to(model.device)
            labels = labels.to(model.device)  # 确保标签在正确的设备上

            optimizer.zero_grad()

            pl = model.forward_normal_for_pl(input[0])
            out = model.forward_aug_with_prompts(input[1].float().cuda())

            pseudo_label = F.softmax(pl, dim=-1)  # / 0.04
            pseudo_label = pseudo_label.argmax(dim=1, keepdim=True)
            pseudo_label = pseudo_label.flatten().cuda()

            # 计算训练精度（在no_grad之外，因为需要model输出）
            _, predicted = torch.max(out, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

            loss = criteria(out.squeeze(), pseudo_label)
            losses.update(loss.item(), input.size(0))  # 更新损失统计

            if i % args.print_freq == 0:
                # 计算当前批次的训练精度
                current_acc = correct_predictions / total_samples * 100 if total_samples > 0 else 0.0
                print(
                    "epoch [{0}/{1}][{2}/{3}]\t"
                    "loss {losses:.4f}\t"
                    "training_acc {training_acc:.2f}\t"
                    "lr {lr:.6e}".format(
                        epoch + 1,
                        args.epochs,
                        i + 1,
                        len(tr_loader),
                        losses=losses.val,  # 使用当前批次损失
                        training_acc=current_acc,
                        lr=optimizer.param_groups[0]["lr"],
                    ))

            loss.backward()
            optimizer.step()
        scheduler.step()

        # 打印epoch总结信息
        epoch_loss = losses.avg
        epoch_acc = correct_predictions / total_samples * 100 if total_samples > 0 else 0.0
        print(f'Epoch [{epoch+1}/{args.epochs}] completed:')
        print(f'  Average Training Loss: {epoch_loss:.4f}')
        print(f'  Training Accuracy: {epoch_acc:.2f}%')

        # 添加学习率信息
        current_lrs = scheduler.get_lr()

        # 智能识别主参数和视觉编码器的学习率
        main_lr = None
        visual_lr = None

        # 遍历所有参数组，根据名称识别
        for i, param_group in enumerate(optimizer.param_groups):
            group_name = param_group.get('name', f'group_{i}')
            if 'visual' in group_name.lower() and i < len(current_lrs):
                visual_lr = current_lrs[i]
            elif 'main' in group_name.lower() or 'adapter' in group_name.lower() or 'prompt' in group_name.lower():
                if i < len(current_lrs):
                    main_lr = current_lrs[i]

        # 如果没有找到，则使用默认策略
        if main_lr is None and len(current_lrs) > 0:
            # 找到最大的学习率作为主参数学习率（通常主参数LR更大）
            main_lr = max(current_lrs)
            visual_lr = min(current_lrs) if len(current_lrs) > 1 else None

        print(f'  Main LR: {main_lr:.6e}')
        if visual_lr is not None:
            print(f'  Visual LR: {visual_lr:.6e}')

        print(f'Evaluation: {epoch}')
        acc = test_prompting(val_loader, model)
        print(f'TOP-1 Accuracy: {acc}')
        all_acc.append(acc)
    print(f'-------------------------------- Best Accuracy: {max(all_acc)} --------------------------------')


def train_lafter_with_text_supervised(args, model, tr_loader, val_loader, text_supervisor):
    """
    有监督版本的LaFTer训练函数，包含文本监督
    使用真实标签进行训练，移除弱增强分支，同时保持text_img_losses
    """

    # 完全复用现有的初始化逻辑
    train_txt_cls(args, model)

    # 初始化WandB监控
    wandb_enabled = hasattr(args, 'enable_wandb') and args.enable_wandb
    config_dict = {
        'dataset': args.dataset_config_file.split('/')[-1].replace('.yaml', ''),
        'text_img_alpha': args.text_img_alpha,
        'learning_rate': args.lr,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'txt_epochs': args.txt_epochs,
        'architecture': args.arch,
        'scheduler': args.scheduler,
        'weight_decay': args.weight_decay,
        'mode': 'supervised_with_text'
    }

    monitor = create_monitor(
        config_dict=config_dict,
        enable_wandb=wandb_enabled,
        project_name=f"LaFTer-Supervised-{config_dict['dataset']}"
    )

    # 打印图像-文本对组合验证（如果启用）
    if hasattr(args, 'print_image_text_pairs') and args.print_image_text_pairs:
        print_image_text_pairs(tr_loader, text_supervisor, model, max_samples=20)

    all_acc = list()

    # 使用增强的训练设置（支持选择性visual encoder解冻）
    if args.enable_visual_unfreezing:
        print("=== Using Enhanced Training Setup with Selective Visual Unfreezing ===")
        print(f"Dropout Configuration: Adapter={args.adapter_dropout}, Projector={args.projector_dropout}, Visual={args.visual_dropout}")
        optimizer, scheduler, criteria, dropout_manager = setup_enhanced_lafter_training_utils(
            args, model,
            enable_visual_unfreezing=True,
            visual_unfreeze_strategy=args.visual_unfreeze_strategy,
            visual_unfreeze_blocks=args.visual_unfreeze_blocks,
            visual_lr=args.visual_lr,
            # Dropout parameters
            adapter_dropout=args.adapter_dropout,
            projector_dropout=args.projector_dropout,
            visual_dropout=args.visual_dropout
        )
    else:
        print("=== Using Standard Training Setup ===")
        print("Note: Dropout parameters are only active with --enable_visual_unfreezing")
        optimizer, scheduler, criteria = setup_lafter_training_utils(args, model)
    batch_time = lossmeter()
    data_time = lossmeter()
    losses = lossmeter()  # 分类损失跟踪器
    text_img_losses = lossmeter()  # 文本监督损失跟踪器
    correct_predictions = 0  # 初始化正确预测计数
    total_samples = 0  # 初始化总样本计数

    print("=== SUPERVISED LEARNING MODE WITH TEXT SUPERVISION ===")
    print(f"Using real labels from dataset + text supervision (alpha={args.text_img_alpha})")
    print("Weak augmentation branch removed - only using strong augmentation")

    for epoch in range(args.epochs):
        print(f'Epoch: {epoch}')
        model.eval()
        model.adapter.train()
        end = time.time()

        # 重置每个epoch的统计信息
        losses.reset()
        text_img_losses.reset()
        correct_predictions = 0
        total_samples = 0

        for i, batch in enumerate(tr_loader):
            data_time.update(time.time() - end)
            batch_time.update(time.time() - end)

            # 有监督学习：使用增强的批次处理函数获取文本
            input, labels, texts = enhance_batch_with_text(batch, text_supervisor, model)

            optimizer.zero_grad()

            # 有监督学习：只使用强增强分支进行训练
            out = model.forward_aug_with_prompts(input[1].float().cuda())

            # 有监督的交叉熵损失计算（使用真实标签）
            cls_loss = criteria(out.squeeze(), labels)

            # 文本监督损失
            text_img_loss = text_supervisor.compute_text_img_loss(model, input[1], texts)

            # 组合损失：(1-α) * cls_loss + α * text_img_loss
            total_loss = (1 - args.text_img_alpha) * cls_loss + args.text_img_alpha * text_img_loss

            # 更新损失统计
            losses.update(cls_loss.item(), input.size(0))
            text_img_losses.update(text_img_loss.item(), input.size(0))

            # 计算训练精度（使用真实标签）
            _, predicted = torch.max(out, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

            # 增强日志输出
            if i % args.print_freq == 0:
                current_acc = correct_predictions / total_samples * 100 if total_samples > 0 else 0.0
                log_msg = ("epoch [{0}/{1}][{2}/{3}]\t"
                          "cls_loss {losses:.4f}\t"
                          "training_acc {training_acc:.2f}\t"
                          "lr {lr:.6e}").format(
                          epoch + 1, args.epochs, i + 1, len(tr_loader),
                          losses=losses.val,
                          training_acc=current_acc,
                          lr=optimizer.param_groups[0]["lr"])

                # 添加文本监督损失日志
                log_msg += f"\ttext_img_loss {text_img_losses.val:.4f}"

                print(log_msg)

            # WandB记录训练步骤（每个batch都记录）
            current_acc = correct_predictions / total_samples * 100 if total_samples > 0 else 0.0
            monitor.log_training_step(
                total_loss=total_loss.item(),
                text_img_loss=text_img_loss.item(),
                train_acc=current_acc,
                epoch=epoch,
                batch_idx=i,
                learning_rate=optimizer.param_groups[0]["lr"]
            )

            total_loss.backward()
            optimizer.step()

        scheduler.step()

        # Epoch总结
        epoch_loss = losses.avg
        epoch_acc = correct_predictions / total_samples * 100 if total_samples > 0 else 0.0
        epoch_text_img_loss = text_img_losses.avg

        print(f'Epoch [{epoch+1}/{args.epochs}] completed:')
        print(f'  Average Classification Loss: {epoch_loss:.4f}')
        print(f'  Training Accuracy: {epoch_acc:.2f}%')
        print(f'  Average Text-Image Loss: {epoch_text_img_loss:.4f}')

        # 添加学习率信息
        current_lrs = scheduler.get_lr()

        # 智能识别主参数和视觉编码器的学习率
        main_lr = None
        visual_lr = None

        # 遍历所有参数组，根据名称识别
        for i, param_group in enumerate(optimizer.param_groups):
            group_name = param_group.get('name', f'group_{i}')
            if 'visual' in group_name.lower() and i < len(current_lrs):
                visual_lr = current_lrs[i]
            elif 'main' in group_name.lower() or 'adapter' in group_name.lower() or 'prompt' in group_name.lower():
                if i < len(current_lrs):
                    main_lr = current_lrs[i]

        # 如果没有找到，则使用默认策略
        if main_lr is None and len(current_lrs) > 0:
            # 找到最大的学习率作为主参数学习率（通常主参数LR更大）
            main_lr = max(current_lrs)
            visual_lr = min(current_lrs) if len(current_lrs) > 1 else None

        print(f'  Main LR: {main_lr:.6e}')
        if visual_lr is not None:
            print(f'  Visual LR: {visual_lr:.6e}')

        # 如果启用了visual encoder unfreezing，额外显示参数统计
        if hasattr(args, 'enable_visual_unfreezing') and args.enable_visual_unfreezing:
            visual_trainable = sum(p.numel() for name, p in model.named_parameters()
                                 if p.requires_grad and 'visual' in name)
            main_trainable = sum(p.numel() for name, p in model.named_parameters()
                               if p.requires_grad and 'visual' not in name)
            total_trainable = visual_trainable + main_trainable
            print(f'  Trainable Params: {total_trainable:,} (Visual: {visual_trainable:,}, Main: {main_trainable:,})')

        # 评估
        print(f'Evaluation: {epoch}')
        acc = test_prompting(val_loader, model)
        print(f'TOP-1 Accuracy: {acc}')
        all_acc.append(acc)

        # WandB记录验证准确率和epoch总结
        monitor.log_validation(val_acc=acc, epoch=epoch)
        monitor.log_epoch_summary(
            avg_total_loss=epoch_loss,
            avg_text_img_loss=epoch_text_img_loss,
            avg_train_acc=epoch_acc,
            val_acc=acc,
            epoch=epoch
        )

    print(f'-------------------------------- Best Accuracy: {max(all_acc)} --------------------------------')

    # 结束WandB监控
    monitor.finish()


def train_lafter_with_text(args, model, tr_loader, val_loader, text_supervisor):
    """
    增强版LaFTer训练函数，包含文本监督
    复用所有现有逻辑，只添加文本监督部分
    """

    # 完全复用现有的初始化逻辑
    train_txt_cls(args, model)

    # 初始化WandB监控
    wandb_enabled = hasattr(args, 'enable_wandb') and args.enable_wandb
    config_dict = {
        'dataset': args.dataset_config_file.split('/')[-1].replace('.yaml', ''),
        'text_img_alpha': args.text_img_alpha,
        'learning_rate': args.lr,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'txt_epochs': args.txt_epochs,
        'architecture': args.arch,
        'scheduler': args.scheduler,
        'weight_decay': args.weight_decay
    }

    monitor = create_monitor(
        config_dict=config_dict,
        enable_wandb=wandb_enabled,
        project_name=f"LaFTer-{config_dict['dataset']}"
    )

    # 打印图像-文本对组合验证（如果启用）
    if hasattr(args, 'print_image_text_pairs') and args.print_image_text_pairs:
        print_image_text_pairs(tr_loader, text_supervisor, model, max_samples=20)

    all_acc = list()

    # 使用增强的训练设置（支持选择性visual encoder解冻）
    if args.enable_visual_unfreezing:
        print("=== Using Enhanced Training Setup with Selective Visual Unfreezing ===")
        print(f"Dropout Configuration: Adapter={args.adapter_dropout}, Projector={args.projector_dropout}, Visual={args.visual_dropout}")
        optimizer, scheduler, criteria, dropout_manager = setup_enhanced_lafter_training_utils(
            args, model,
            enable_visual_unfreezing=True,
            visual_unfreeze_strategy=args.visual_unfreeze_strategy,
            visual_unfreeze_blocks=args.visual_unfreeze_blocks,
            visual_lr=args.visual_lr,
            # Dropout parameters
            adapter_dropout=args.adapter_dropout,
            projector_dropout=args.projector_dropout,
            visual_dropout=args.visual_dropout
        )
    else:
        print("=== Using Standard Training Setup ===")
        print("Note: Dropout parameters are only active with --enable_visual_unfreezing")
        optimizer, scheduler, criteria = setup_lafter_training_utils(args, model)
    batch_time = lossmeter()
    data_time = lossmeter()
    losses = lossmeter()  # 分类损失跟踪器
    text_img_losses = lossmeter()  # 文本监督损失跟踪器
    correct_predictions = 0  # 初始化正确预测计数
    total_samples = 0  # 初始化总样本计数

    print(f"Text supervision enabled with alpha={args.text_img_alpha}")

    for epoch in range(args.epochs):
        print(f'Epoch: {epoch}')
        model.eval()
        model.adapter.train()
        end = time.time()

        # 重置每个epoch的统计信息
        losses.reset()
        text_img_losses.reset()
        correct_predictions = 0
        total_samples = 0

        for i, batch in enumerate(tr_loader):
            data_time.update(time.time() - end)
            batch_time.update(time.time() - end)

            # 增量修改：使用增强的批次处理函数获取文本
            input, labels, texts = enhance_batch_with_text(batch, text_supervisor, model)

            optimizer.zero_grad()

            # 完全复用现有的前向传播逻辑
            pl = model.forward_normal_for_pl(input[0])
            out = model.forward_aug_with_prompts(input[1].float().cuda())
            pseudo_label = F.softmax(pl, dim=-1)
            pseudo_label = pseudo_label.argmax(dim=1, keepdim=True)
            pseudo_label = pseudo_label.flatten().cuda()

            # 现有的分类损失计算（使用伪标签）
            cls_loss = criteria(out.squeeze(), pseudo_label)

            # 增量添加：文本监督损失
            text_img_loss = text_supervisor.compute_text_img_loss(model, input[1], texts)

            # 组合损失：(1-α) * cls_loss + α * text_img_loss
            total_loss = (1 - args.text_img_alpha) * cls_loss + args.text_img_alpha * text_img_loss

            # 更新损失统计
            losses.update(cls_loss.item(), input.size(0))
            text_img_losses.update(text_img_loss.item(), input.size(0))

            # 复用现有的准确率计算和日志逻辑
            _, predicted = torch.max(out, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

            # 增量修改：增强日志输出
            if i % args.print_freq == 0:
                current_acc = correct_predictions / total_samples * 100 if total_samples > 0 else 0.0
                log_msg = ("epoch [{0}/{1}][{2}/{3}]\t"
                          "loss {losses:.4f}\t"
                          "training_acc {training_acc:.2f}\t"
                          "lr {lr:.6e}").format(
                          epoch + 1, args.epochs, i + 1, len(tr_loader),
                          losses=losses.val,
                          training_acc=current_acc,
                          lr=optimizer.param_groups[0]["lr"])

                # 添加文本监督损失日志
                log_msg += f"\ttext_img_loss {text_img_losses.val:.4f}"

                print(log_msg)

            # WandB记录训练步骤（每个batch都记录）
            current_acc = correct_predictions / total_samples * 100 if total_samples > 0 else 0.0
            monitor.log_training_step(
                total_loss=total_loss.item(),
                text_img_loss=text_img_loss.item(),
                train_acc=current_acc,
                epoch=epoch,
                batch_idx=i,
                learning_rate=optimizer.param_groups[0]["lr"]
            )

            total_loss.backward()
            optimizer.step()

        scheduler.step()

        # 复用现有的epoch总结逻辑
        epoch_loss = losses.avg
        epoch_acc = correct_predictions / total_samples * 100 if total_samples > 0 else 0.0
        epoch_text_img_loss = text_img_losses.avg

        print(f'Epoch [{epoch+1}/{args.epochs}] completed:')
        print(f'  Average Training Loss: {epoch_loss:.4f}')
        print(f'  Training Accuracy: {epoch_acc:.2f}%')
        print(f'  Average Text-Image Loss: {epoch_text_img_loss:.4f}')

        # 添加学习率信息
        current_lrs = scheduler.get_lr()

        # 智能识别主参数和视觉编码器的学习率
        main_lr = None
        visual_lr = None

        # 遍历所有参数组，根据名称识别
        for i, param_group in enumerate(optimizer.param_groups):
            group_name = param_group.get('name', f'group_{i}')
            if 'visual' in group_name.lower() and i < len(current_lrs):
                visual_lr = current_lrs[i]
            elif 'main' in group_name.lower() or 'adapter' in group_name.lower() or 'prompt' in group_name.lower():
                if i < len(current_lrs):
                    main_lr = current_lrs[i]

        # 如果没有找到，则使用默认策略
        if main_lr is None and len(current_lrs) > 0:
            # 找到最大的学习率作为主参数学习率（通常主参数LR更大）
            main_lr = max(current_lrs)
            visual_lr = min(current_lrs) if len(current_lrs) > 1 else None

        print(f'  Main LR: {main_lr:.6e}')
        if visual_lr is not None:
            print(f'  Visual LR: {visual_lr:.6e}')

        # 如果启用了visual encoder unfreezing，额外显示参数统计
        if hasattr(args, 'enable_visual_unfreezing') and args.enable_visual_unfreezing:
            visual_trainable = sum(p.numel() for name, p in model.named_parameters()
                                 if p.requires_grad and 'visual' in name)
            main_trainable = sum(p.numel() for name, p in model.named_parameters()
                               if p.requires_grad and 'visual' not in name)
            total_trainable = visual_trainable + main_trainable
            print(f'  Trainable Params: {total_trainable:,} (Visual: {visual_trainable:,}, Main: {main_trainable:,})')

        # 完全复用现有的评估逻辑
        print(f'Evaluation: {epoch}')
        acc = test_prompting(val_loader, model)
        print(f'TOP-1 Accuracy: {acc}')
        all_acc.append(acc)

        # WandB记录验证准确率和epoch总结
        monitor.log_validation(val_acc=acc, epoch=epoch)
        monitor.log_epoch_summary(
            avg_total_loss=epoch_loss,
            avg_text_img_loss=epoch_text_img_loss,
            avg_train_acc=epoch_acc,
            val_acc=acc,
            epoch=epoch
        )

    print(f'-------------------------------- Best Accuracy: {max(all_acc)} --------------------------------')

    # 结束WandB监控
    monitor.finish()


def test_img_classifier(args, model, teloader):
    """
    用已经训练好的图像分类器，在测试集上跑一次测试返回 top-1 精度。
    处理图像输入而不是文本prompt。
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(teloader):
            # 从batch中提取图像和标签
            images = batch["img"].to(model.device)  # 确保图像在正确的设备上
            labels = batch["label"].to(model.device)  # 确保标签在相同的设备上
            
            # 需要确保模型有正确的评估方法
            try:
                # 尝试使用模型特定的评估方法(如果存在)
                outputs = model.forward_normal_for_pl(images)  # 或model.eval_img_clas(images)
            except AttributeError:
                # 针对数据类型不匹配问题的处理
                # 这里我们假设模型有一个特定的forward方法需要调用
                # 您可能需要根据实际模型的实现来调整
                if hasattr(model, 'text_features') and model.text_features.dtype != images.dtype:
                    # 如果text_features是half类型，将images也转为half
                    if model.text_features.dtype == torch.float16:
                        images = images.half()
                    # 如果images是half类型，将text_features转为float
                    elif images.dtype == torch.float16:
                        model.text_features = model.text_features.float()
                
                # 使用常规forward
                outputs = model(images)
            
            # 计算预测结果
            preds = outputs.argmax(dim=1)
            
            # 确保两个张量在同一设备上进行比较
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return 100.0 * correct / total



def train_visual_only_simple(args, model, train_loader, val_loader):
    """
    Simple visual-only training function for frozen CLIP + trainable classifier
    """
    import torch.nn as nn
    import torch.optim as optim

    print("=== Visual-Only Training ===")
    print("Frozen CLIP encoder + trainable 2-layer classifier")

    # Setup optimizer (only for classifier parameters)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=getattr(args, 'head_lr', 1e-3)
    )

    # Setup loss function
    criterion = nn.CrossEntropyLoss()

    # WandB monitor setup
    wandb_enabled = hasattr(args, 'enable_wandb') and args.enable_wandb
    config_payload = _sanitize_wandb_config(args)
    dataset_tag = "unknown"
    if getattr(args, 'dataset_config_file', None):
        dataset_tag = Path(args.dataset_config_file).stem
    elif hasattr(train_loader.dataset, 'dataset_name'):
        dataset_tag = getattr(train_loader.dataset, 'dataset_name')
    monitor = create_monitor(
        config_dict=config_payload,
        enable_wandb=wandb_enabled,
        project_name=f"LaFTer-VisualOnly-{dataset_tag}"
    )

    # Training loop
    all_acc = list()
    best_val = 0.0

    for epoch in range(args.mm_epochs):
        print(f'Epoch: {epoch}')
        model.train()

        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for i, batch in enumerate(train_loader):
            images = batch["img"]
            labels = batch["label"]

            # Handle dual-view images - take first view
            if isinstance(images, list):
                images = images[0]
            elif hasattr(images, '__len__') and len(images.shape) == 5:  # [B, 2, C, H, W]
                images = images[:, 0]  # Take first view

            images = images.to(model.device)
            labels = labels.to(model.device)

            optimizer.zero_grad()

            # Forward pass - visual-only mode only needs images
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

            if i % getattr(args, 'print_freq', 50) == 0:
                current_acc = correct_predictions / total_samples * 100 if total_samples > 0 else 0.0
                print(
                    "epoch [{0}/{1}][{2}/{3}]\t"
                    "loss {loss:.4f}\t"
                    "training_acc {acc:.2f}%".format(
                        epoch + 1,
                        args.mm_epochs,
                        i + 1,
                        len(train_loader),
                        loss=running_loss/(i+1),
                        acc=current_acc
                    )
                )

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                images = batch["img"]
                labels = batch["label"]

                # Handle dual-view images
                if isinstance(images, list):
                    images = images[0]
                elif hasattr(images, '__len__') and len(images.shape) == 5:
                    images = images[:, 0]

                images = images.to(model.device)
                labels = labels.to(model.device)

                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = 100. * val_correct / val_total
        train_acc = 100. * correct_predictions / total_samples
        all_acc.append(val_acc)
        best_val = max(best_val, val_acc)

        print(f'Epoch [{epoch+1}/{args.mm_epochs}] completed:')
        epoch_loss = running_loss/len(train_loader)
        print(f'  Average Training Loss: {epoch_loss:.4f}')
        print(f'  Training Accuracy: {train_acc:.2f}%')
        print(f'  Validation Accuracy: {val_acc:.2f}%')
        print(f'TOP-1 Accuracy: {val_acc:.2f}%')

        monitor.log_epoch_summary(
            avg_total_loss=epoch_loss,
            avg_text_img_loss=0.0,
            avg_train_acc=train_acc,
            val_acc=val_acc,
            epoch=epoch
        )

    monitor.finish()
    print(f'-------------------------------- Best Accuracy: {best_val:.2f}% --------------------------------')


def train_multimodal_dassl(args):
    """
    New multimodal training pipeline using:
    - Visual encoder: Dassl-style CLIP (via src.clip_vision_encoder + utils.model_utils.load_clip_to_cpu)
    - Text encoder: roberta-large (default), or a local HF dir via --bert_dir
    - Input: pre-built image-text pairs (JSONL) without templates/descriptions
    """
    print("=== Starting Multimodal Training (Dassl-CLIP + Roberta) ===")

    wandb_enabled = hasattr(args, 'enable_wandb') and args.enable_wandb
    run_config = _sanitize_wandb_config(args)

    # Seed for reproducibility (simple & robust, without forcing strict deterministic kernels)
    try:
        print(f"Setting fixed seed: {getattr(args, 'seed', 7777)}")
        set_random_seed(getattr(args, 'seed', 7777))
        import torch
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    except Exception as _seed_e:
        print(f"Warning: failed to set seed: {_seed_e}")

    import os
    from torch.utils.data import DataLoader

    # Import multimodal components
    from trainers.mm_trainer import MMTrainer
    from utils.pairs_dataset import ImageTextPairs, collate_fn
    # Reuse EXACT SAME image transforms as Dassl/LaFTer visual pipeline
    # - Training: transform_default_clip_weakly_aug (the same first-view used in visual_only)
    # - Validation: te_transform
    # This aligns interpolation (Bicubic), RGB conversion and normalization with Dassl.
    from utils.model_utils import te_transform, transform_default_clip_weakly_aug

    # Validate arguments
    if not args.train_pairs:
        raise ValueError("train_pairs must be specified for multimodal pipeline")
    if not args.val_pairs:
        raise ValueError("val_pairs must be specified for multimodal pipeline")

    print(f"Train pairs: {args.train_pairs}")
    print(f"Val pairs: {args.val_pairs}")
    print(f"Text encoder (HF): {args.bert_dir or args.bert_model_name}")
    print(f"Vision encoder (CLIP): {args.arch}")
    print(f"Text adapter enabled: {getattr(args, 'use_adapter', False)}")
    if getattr(args, "use_adapter", False):
        print(f"  Adapter type: {args.adapter_type} | reduction: {args.adapter_reduction_factor} | dropout: {args.adapter_dropout}")
        print(f"  Adapter LR: {args.lr_adapter if args.lr_adapter is not None else args.bert_lr}")

    # Setup tokenizer (local cache only if offline)
    from transformers import AutoTokenizer
    cache_dir = os.getenv("TRANSFORMERS_CACHE") or os.getenv("HF_HOME")
    local_only = str(os.getenv("TRANSFORMERS_OFFLINE", "0")).lower() in ("1", "true", "yes") \
        or str(os.getenv("HF_FORCE_LOCAL", "0")).lower() in ("1", "true", "yes")
    tokenizer = AutoTokenizer.from_pretrained(
        args.bert_dir or args.bert_model_name,
        cache_dir=cache_dir,
        local_files_only=local_only
    )

    # Datasets (use Dassl-aligned transforms)
    # Note: visual_only path in LaFTer uses DataManager with TwoCropsTransform,
    # but the training loop actually consumes the first (weak) view. Here we
    # pass that weak view transform directly for a 1:1 equivalence.
    train_dataset = ImageTextPairs(
        pairs_file=args.train_pairs,
        img_root=args.root,
        tokenizer=tokenizer,
        transform=transform_default_clip_weakly_aug
    )
    val_dataset = ImageTextPairs(
        pairs_file=args.val_pairs,
        img_root=args.root,
        tokenizer=tokenizer,
        transform=te_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.mm_batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.mm_batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_fn
    )

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")

    # Create multimodal model: force CLIP vision encoder, Roberta-large text
    from src.mm_model import MultiModalModel
    adapter_trainable_override = True if getattr(args, "use_adapter", False) else None

    model = MultiModalModel(
        num_classes=8,
        encoder_type='clip',
        vision_model_name=args.arch,
        bert_model_dir=args.bert_dir or None,
        bert_model_name=args.bert_model_name or 'roberta-large',
        freeze_bert=args.freeze_bert,
        freeze_vision=getattr(args, "freeze_vision", False),
        fusion_dim=args.fusion_dim,
        fusion_type=args.fusion_type,
        use_adapter=getattr(args, "use_adapter", False),
        adapter_type=getattr(args, "adapter_type", "pfeiffer"),
        adapter_reduction_factor=getattr(args, "adapter_reduction_factor", 16),
        adapter_dropout=getattr(args, "adapter_dropout", 0.0),
        adapter_trainable=adapter_trainable_override
    )

    dataset_tag = Path(args.train_pairs).stem if args.train_pairs else "unknown"
    trainer = MMTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        dataset_name=getattr(args, 'dataset_name', 'Emotion6'),
        vit_lr=args.vit_lr,
        bert_lr=args.bert_lr,
        head_lr=args.head_lr,
        lr_adapter=getattr(args, "lr_adapter", None),
        epochs=args.mm_epochs,
        mixed_precision=False,
        save_dir=getattr(args, 'mm_save_dir', 'checkpoints/mm'),
        print_freq=getattr(args, 'print_freq', 50),
        log_wandb=wandb_enabled,
        project_name=f"LaFTer-Multimodal-{dataset_tag}",
        monitor_modal_losses=True,
        modal_loss_freq=getattr(args, 'modal_loss_freq', getattr(args, 'print_freq', 50))
    )
    if wandb_enabled and hasattr(trainer, "wandb"):
        try:
            trainer.wandb.config.update(run_config, allow_val_change=True)
        except Exception as wandb_update_error:
            print(f"Warning: Failed to update wandb config: {wandb_update_error}")

    print("Starting multimodal training...")
    trainer.train()

    print("=== Multimodal training completed ===")


def main(args):
    # Check if using multimodal pipeline (new Dassl-CLIP + Roberta path)
    if hasattr(args, 'pipeline') and args.pipeline == 'multimodal':
        return train_multimodal_dassl(args)

    cfg = setup_cfg(args)
    cfg.DATALOADER.TRAIN_X.BATCH_SIZE = args.batch_size
    cfg.DATALOADER.TEST.BATCH_SIZE = args.batch_size
    cfg.SEED = args.seed

    dataset_name = cfg.DATASET.NAME
    setup_txt_epochs(args, dataset_name)

    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)
    print_args(args, cfg)
    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True
    trainer = build_trainer(cfg)
    model = trainer.model
    model.args = args

    train_loader = trainer.train_loader_x
    test_loader = trainer.test_loader

    # 增量添加：文本监督器初始化
    text_supervisor = None
    if args.use_text_supervision:
        print("Initializing text supervisor...")
        # 从数据集配置文件中获取数据集名称
        dataset_name = cfg.DATASET_NAME if hasattr(cfg, 'DATASET_NAME') else cfg.DATASET.NAME
        text_supervisor = TextSupervisor(args.text_csv_path, args.text_img_alpha, dataset_name)
        print(f"Text supervision enabled with alpha={args.text_img_alpha} for dataset {dataset_name}")

        # 验证文本覆盖率（暂时注释掉，因为数据集结构问题）
        # train_image_paths = [item.impath for item in train_loader.dataset]
        # coverage_stats = text_supervisor.validate_coverage(train_image_paths)
        # if coverage_stats['coverage_rate'] < 0.8:
        #     print(f"Warning: Low text coverage rate ({coverage_stats['coverage_rate']:.2%})")
        #     print(f"Matched: {coverage_stats['matched_images']}/{coverage_stats['total_images']}")
        # else:
        #     print(f"Text coverage rate: {coverage_stats['coverage_rate']:.2%}")
        print("Text coverage validation skipped")

    # —— 文本分类器训练和测试 ——
    if args.txt_cls in ['cls_only', 'templates_only'] or (args.txt_cls == 'lafter' and args.skip_finetune):
        mode_desc = "仅文本分类器" if args.skip_finetune else args.txt_cls
        print(f"只进行文本分类器训练 (mode={mode_desc})，跳过图像微调")
        # 1) 训练文本分类器
        train_txt_cls(args, model)
        # 2) 在测试集上评估文本分类器
        acc = test_img_classifier(args, model, test_loader)
        print(f"[Text-CLS] Test Top-1 Accuracy: {acc:.2f}%")
        return

    # —— Visual-Only 优先处理 ——
    if hasattr(args, 'visual_only') and args.visual_only:
        # Visual-only mode: setup dataset and train visual-only model
        print("=== Visual-Only Training Mode ===")
        print("Using frozen CLIP encoder + trainable classifier")

        # Setup datasets using same infrastructure as original LaFTer
        cfg = setup_cfg(args)
        cfg.DATALOADER.TRAIN_X.BATCH_SIZE = getattr(args, 'mm_batch_size', args.batch_size)
        cfg.DATALOADER.TEST.BATCH_SIZE = getattr(args, 'mm_batch_size', args.batch_size)

        trainer = build_trainer(cfg)
        train_loader = trainer.train_loader_x
        val_loader = trainer.test_loader
        model.args = args

        print(f"Dataset: {cfg.DATASET.NAME}")
        print(f"Train dataset size: {len(train_loader.dataset)}")
        print(f"Val dataset size: {len(val_loader.dataset)}")

        # Get number of classes from dataset
        num_classes = train_loader.dataset.num_classes if hasattr(train_loader.dataset, 'num_classes') else 8
        print(f"Number of classes: {num_classes}")

        # Create visual-only model (minimal parameters for clean interface)
        from src.mm_model import MultiModalModel
        visual_model = MultiModalModel(
            num_classes=num_classes,
            vision_model_name=args.arch,
            visual_only=True,
            dropout=0.1
        )

        # Train using visual-only function
        train_visual_only_simple(args, visual_model, train_loader, val_loader)
        print("=== Visual-Only Training Completed ===")
    # —— 图像微调流程 ——
    elif args.txt_cls == 'lafter' and not args.skip_finetune:
        if args.supervised_mode:
            # 有监督学习模式
            if args.use_text_supervision and text_supervisor:
                print("Starting SUPERVISED LaFTer training with text supervision...")
                train_lafter_with_text_supervised(args, model, train_loader, test_loader, text_supervisor)
            else:
                print("Error: Supervised mode requires text supervision. Please add --use_text_supervision flag.")
                return
        else:
            # 原始无监督学习模式
            if args.use_text_supervision and text_supervisor:
                print("Starting LaFTer training with text supervision...")
                train_lafter_with_text(args, model, train_loader, test_loader, text_supervisor)
            else:
                print("Starting standard LaFTer training...")
                train_lafter(args, model, train_loader, test_loader)
    elif args.zero_shot:
        zero_shot(model, test_loader)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="", help="path to dataset")
    parser.add_argument("--dataset_name", type=str, default="Emotion6",
                        help="Dataset name for binary classification mapping (Emotion6, FI_new, Emoset, etc.)")
    parser.add_argument("--output-dir", type=str, default="", help="output directory")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="checkpoint directory (from which the training resumes)",
    )
    parser.add_argument(
        "--seed", type=int, default=7777, help="only positive value enables a fixed seed"
    )
    parser.add_argument(
        "--print_freq", type=int, default=10, help="only positive value enables a fixed seed"
    )
    parser.add_argument(
        "--source-domains", type=str, nargs="+", help="source domains for DA/DG"
    )
    parser.add_argument(
        "--target-domains", type=str, nargs="+", help="target domains for DA/DG"
    )
    parser.add_argument(
        "--transforms", type=str, nargs="+", help="data augmentation methods"
    )
    parser.add_argument(
        "--config-file", type=str, default="", help="path to config file"
    )
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="",
        help="path to config file for dataset setup",
    )
    parser.add_argument("--trainer", type=str, default="", help="name of trainer")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="",
        help="load model from this directory for eval-only mode",
    )
    parser.add_argument(
        "--load-epoch", type=int, help="load model weights at this epoch for evaluation"
    )
    parser.add_argument(
        "--no-train", action="store_true", help="do not call trainer.train()"
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    parser.add_argument('--exp-name', type=str, required=False)
    parser.add_argument('--scheduler', default='cosine')
    parser.add_argument('--scheduler-epochs', type=int, default=15)
    parser.add_argument('--scheduler-gamma', type=float, default=0.5)
    parser.add_argument('--weight-decay', type=float, default=0.001)
    parser.add_argument('--acc-batches', type=int, default=1)
    parser.add_argument('--arch', type=str, default='ViT-B/32', required=False)
    parser.add_argument('--encoder-type', type=str, default='vit', choices=['vit', 'clip'],
                        help='Vision encoder type: vit or clip')
    parser.add_argument('--gpt_prompts', action='store_true')
    parser.add_argument('--text_prompts', action='store_true')
    parser.add_argument('--zero_shot', action='store_true')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--txt_cls', type=str, default='lafter', choices=['cls_only',
                                                                                   'templates_only', 'lafter', 'zero_shot'])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--txt_epochs', type=int, default=1000)
    parser.add_argument('--logfolder', default='logs', type=str)
    parser.add_argument('--skip_finetune', action='store_true',
                help='只训练文本分类器，跳过图像微调步骤') 
    parser.add_argument('--milestones', type=int, nargs='+', default=[15, 25, 35],
                help='学习率衰减的epoch节点，例如: --milestones 10 20 30')
    parser.add_argument('--print_image_text_pairs', action='store_true',
                       help='在图像微调前打印图像-文本对组合以验证匹配正确性')
    parser.add_argument('--use_text_supervision', action='store_true',
                       help='Enable text supervision using losstextimg')
    parser.add_argument('--enable_wandb', action='store_true',
                       help='Enable WandB monitoring for training metrics')
    parser.add_argument('--text_img_alpha', type=float, default=0.6,
                       help='Weight for text-image loss (default: 0.6)')
    parser.add_argument('--text_csv_path', type=str,
                       default='D:\\narracap_extended_Emotion6.csv',
                       help='Path to CSV file with image-text pairs')
    parser.add_argument('--supervised_mode', action='store_true',
                       help='Enable supervised learning mode using real labels instead of pseudo-labels')

    # Visual encoder选择性解冻参数
    parser.add_argument('--enable_visual_unfreezing', action='store_true',
                       help='Enable selective visual encoder parameter unfreezing')
    parser.add_argument('--visual_unfreeze_strategy', type=str, default='last_blocks',
                       choices=['last_blocks', 'attention_heads', 'layer_norms', 'projection_head', 'custom'],
                       help='Strategy for visual encoder parameter unfreezing')
    parser.add_argument('--visual_unfreeze_blocks', type=int, default=2,
                       help='Number of last transformer blocks to unfreeze (for last_blocks strategy)')
    parser.add_argument('--visual_lr', type=float, default=1e-5,
                       help='Learning rate for visual encoder parameters (default: 1e-5)')

    # Dropout参数 (仅在启用--enable_visual_unfreezing时生效)
    parser.add_argument('--adapter_dropout', type=float, default=0.2,
                       help='Dropout probability for adapter layers (default: 0.2, only active with --enable_visual_unfreezing)')
    parser.add_argument('--projector_dropout', type=float, default=0.1,
                       help='Dropout probability for projector layers (default: 0.1, only active with --enable_visual_unfreezing)')
    parser.add_argument('--visual_dropout', type=float, default=0.15,
                       help='Dropout probability for visual encoder layers (default: 0.15, only active with --enable_visual_unfreezing)')

      # Multimodal pipeline arguments
    parser.add_argument('--pipeline', type=str, default='lafter',
                        choices=['lafter', 'multimodal'],
                        help='Training pipeline: lafter (original) or multimodal (BERT+ViT fusion)')
    parser.add_argument('--train_pairs', type=str, default='',
                        help='Path to training pairs JSONL file for multimodal pipeline')
    parser.add_argument('--val_pairs', type=str, default='',
                        help='Path to validation pairs JSONL file for multimodal pipeline')
    parser.add_argument('--bert_dir', type=str, default='',
                        help='Path to fine-tuned BERT model directory for multimodal pipeline')
    parser.add_argument('--bert_model_name', type=str, default='roberta-large',
                        help='HuggingFace model name for text encoder (default: roberta-large)')
    parser.add_argument('--freeze_bert', action='store_true', default=False,
                        help='Freeze BERT parameters in multimodal training')
    parser.add_argument('--freeze_vision', action='store_true', default=False,
                        help='Freeze vision encoder parameters in multimodal training')
    parser.add_argument('--vit_lr', type=float, default=3e-5,
                        help='Learning rate for ViT encoder in multimodal training')
    parser.add_argument('--bert_lr', type=float, default=1e-5,
                        help='Learning rate for BERT encoder in multimodal training')
    parser.add_argument('--head_lr', type=float, default=1e-3,
                        help='Learning rate for fusion head in multimodal training')
    parser.add_argument('--fusion_dim', type=int, default=768,
                        help='Hidden dimension for fusion MLP in multimodal training')
    parser.add_argument('--fusion_type', type=str, default='standard',
                        choices=['standard', 'lightweight'],
                        help='Type of fusion head: standard (2-layer MLP) or lightweight')
    parser.add_argument('--mm_save_dir', type=str, default='checkpoints/mm',
                        help='Output directory for multimodal model checkpoints')
    parser.add_argument('--mm_epochs', type=int, default=20,
                        help='Number of epochs for multimodal training')
    parser.add_argument('--mm_batch_size', type=int, default=32,
                        help='Batch size for multimodal training')
    parser.add_argument('--use_adapter', action='store_true',
                        help='Enable AdapterHub adapters inside the text encoder for multimodal training')
    parser.add_argument('--adapter_type', type=str, default='pfeiffer',
                        choices=['pfeiffer', 'houlsby'],
                        help='Adapter architecture to attach to the text encoder')
    parser.add_argument('--adapter_reduction_factor', type=int, default=16,
                        help='Reduction factor (bottleneck size) for adapter layers')
    parser.add_argument('--lr_adapter', type=float, default=None,
                        help='Learning rate for adapter parameters (falls back to --bert_lr when unset)')
    parser.add_argument('--lr_head', type=float, default=None,
                        help='Optional alias for --head_lr when scripting convenience is needed')

    # Pure visual classification mode
    parser.add_argument('--visual_only', action='store_true',
                        help='Use frozen CLIP visual encoder + trainable linear classifier')

    args = parser.parse_args()

    # Allow shell scripts to pass --lr_head as an alias to --head_lr
    if getattr(args, "lr_head", None) is not None:
        args.head_lr = args.lr_head

    main(args)
