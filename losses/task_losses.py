"""
任务损失函数：回归 + 分类
"""

import torch
import torch.nn as nn


class TaskLoss(nn.Module):
    """统一的任务损失"""

    def __init__(self, task_type='regression', num_classes=1):
        super().__init__()
        self.task_type = task_type

        if task_type == 'regression':
            self.loss_fn = nn.SmoothL1Loss()
        elif task_type == 'classification':
            self.loss_fn = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unknown task_type: {task_type}")

    def forward(self, logits, labels):
        if self.task_type == 'regression':
            labels = labels.float()
            if labels.dim() == 1:
                labels = labels.unsqueeze(-1)
            return self.loss_fn(logits.squeeze(-1), labels.squeeze(-1))
        else:
            return self.loss_fn(logits, labels.long().squeeze())
