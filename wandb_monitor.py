#!/usr/bin/env python3
"""
LaFTer WandB监控模块
专门监控total_losses, text_img_losses, val_acc, train_acc四个关键指标
"""

import wandb
import torch
from typing import Optional, Dict, Any
import numpy as np

class LaFTerWandBMonitor:
    """LaFTer训练监控器"""

    def __init__(self,
                 project_name: str = "LaFTer-Experiment",
                 run_name: Optional[str] = None,
                 config: Optional[Dict[str, Any]] = None,
                 enable: bool = True):
        """
        初始化WandB监控

        Args:
            project_name: WandB项目名称
            run_name: 运行名称，如果为None则自动生成
            config: 配置信息字典
            enable: 是否启用监控
        """
        self.enable = enable
        self.step_count = 0

        if not enable:
            print("WandB监控已禁用")
            return

        # 初始化wandb
        wandb.init(
            project=project_name,
            name=run_name,
            config=config or {},
            settings=wandb.Settings(silent=True)
        )

        print(f"WandB监控已启动 - 项目: {project_name}, 运行: {wandb.run.name}")

    def log_training_step(self,
                      total_loss: float,
                      text_img_loss: float,
                      train_acc: float,
                      epoch: int = 0,
                      batch_idx: int = 0,
                      learning_rate: Optional[float] = None,
                      additional_metrics: Optional[Dict[str, float]] = None):
        """
        记录训练步骤

        Args:
            total_loss: 总损失
            text_img_loss: 文本-图像损失
            train_acc: 训练准确率
            epoch: 当前epoch
            batch_idx: 当前batch索引
            learning_rate: 学习率
            additional_metrics: 额外的指标
        """
        if not self.enable:
            return

        # 构建基础指标
        metrics = {
            'total_loss': total_loss,
            'text_img_loss': text_img_loss,
            'train_acc': train_acc,
            'epoch': epoch,
            'batch': batch_idx
        }

        # 添加学习率
        if learning_rate is not None:
            metrics['learning_rate'] = learning_rate

        # 添加额外指标
        if additional_metrics:
            metrics.update(additional_metrics)

        # 记录到wandb
        wandb.log(metrics, step=self.step_count)
        self.step_count += 1

        # 控制台输出（可选）
        if batch_idx % 50 == 0:  # 每50个batch打印一次
            print(f"Step {self.step_count}: "
                  f"Total Loss={total_loss:.4f}, "
                  f"Text-Img Loss={text_img_loss:.4f}, "
                  f"Train Acc={train_acc:.2f}%")

    def log_validation(self,
                     val_acc: float,
                     epoch: int = 0,
                     additional_metrics: Optional[Dict[str, float]] = None):
        """
        记录验证结果

        Args:
            val_acc: 验证准确率
            epoch: 当前epoch
            additional_metrics: 额外的指标
        """
        if not self.enable:
            return

        # 构建验证指标
        val_metrics = {
            'val_acc': val_acc,
            'epoch': epoch
        }

        # 添加额外指标
        if additional_metrics:
            val_metrics.update(additional_metrics)

        # 记录到wandb
        wandb.log(val_metrics, step=self.step_count)

        print(f"Epoch {epoch} Validation: Val Acc={val_acc:.2f}%")

    def log_epoch_summary(self,
                       avg_total_loss: float,
                       avg_text_img_loss: float,
                       avg_train_acc: float,
                       val_acc: float,
                       epoch: int,
                       additional_metrics: Optional[Dict[str, float]] = None):
        """
        记录epoch汇总信息

        Args:
            avg_total_loss: 平均总损失
            avg_text_img_loss: 平均文本-图像损失
            avg_train_acc: 平均训练准确率
            val_acc: 验证准确率
            epoch: 当前epoch
            additional_metrics: 额外的指标
        """
        if not self.enable:
            return

        # 构建epoch汇总指标
        summary = {
            'epoch_summary/avg_total_loss': avg_total_loss,
            'epoch_summary/avg_text_img_loss': avg_text_img_loss,
            'epoch_summary/avg_train_acc': avg_train_acc,
            'epoch_summary/val_acc': val_acc,
            'epoch': epoch
        }

        # 添加额外指标
        if additional_metrics:
            for key, value in additional_metrics.items():
                summary[f'epoch_summary/{key}'] = value

        # 记录到wandb
        wandb.log(summary, step=self.step_count)

        print(f"\n=== Epoch {epoch} Summary ===")
        print(f"Avg Total Loss: {avg_total_loss:.4f}")
        print(f"Avg Text-Img Loss: {avg_text_img_loss:.4f}")
        print(f"Avg Train Acc: {avg_train_acc:.2f}%")
        print(f"Val Acc: {val_acc:.2f}%")
        print("=" * 30 + "\n")

    def log_model_info(self, model_config: Dict[str, Any]):
        """记录模型配置信息"""
        if not self.enable:
            return

        wandb.config.update(model_config)

    def finish(self):
        """完成监控"""
        if self.enable:
            print("WandB监控已结束")
            wandb.finish()


# 便捷函数
def create_monitor(config_dict: Dict[str, Any] = None,
                enable_wandb: bool = True,
                project_name: str = "LaFTer-Experiment") -> LaFTerWandBMonitor:
    """
    创建监控器的便捷函数

    Args:
        config_dict: 配置字典
        enable_wandb: 是否启用wandb
        project_name: 项目名称

    Returns:
        LaFTerWandBMonitor实例
    """
    return LaFTerWandBMonitor(
        project_name=project_name,
        config=config_dict,
        enable=enable_wandb
    )