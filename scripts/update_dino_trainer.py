dino_trainer_path = "src/ag_foundation/training/dino_trainer.py"
with open(dino_trainer_path, encoding="utf-8") as f:
    dino_trainer_content = f.read()

# 1. Add DINOForwardWrapper
wrapper_code = """
import torch.nn as nn

class _DINOForwardWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, student_views, teacher_views, student_temperature, gram_anchor_max_tokens):
        student_outputs = self.model.forward_student_views(student_views)
        teacher_outputs = self.model.forward_teacher_views(teacher_views)
        student_dense_views = None
        teacher_dense_views = None
        if self.model.gram_anchor_weight > 0.0:
            student_dense_views = self.model.student_dense_views(student_views[: len(teacher_views)])
            teacher_dense_views = self.model.teacher_dense_views(teacher_views)
        return self.model.dino_v3_loss(
            student_outputs,
            teacher_outputs,
            student_dense_views=student_dense_views,
            teacher_dense_views=teacher_dense_views,
            student_temperature=student_temperature,
            gram_anchor_weight=self.model.gram_anchor_weight,
            gram_anchor_max_tokens=gram_anchor_max_tokens,
        )

"""
if "class _DINOForwardWrapper" not in dino_trainer_content:
    dino_trainer_content = dino_trainer_content.replace(
        "class DINOTrainer:",
        wrapper_code + "\nclass DINOTrainer:"
    )

# 2. Update __init__
init_old = """
        self.run_config = dict(run_config or {})
        self.device = torch.device(device) if device is not None else torch.device(select_torch_device())
        self.model = self.model.to(self.device)
        self._trainable_parameter_items = [
            (name, parameter)
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        ]
"""
init_new = """
        import os
        self.run_config = dict(run_config or {})
        self.device = torch.device(device) if device is not None else torch.device(select_torch_device())
        
        self.raw_model = model.to(self.device)
        self.ddp_wrapper = _DINOForwardWrapper(self.raw_model)
        
        self.is_distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
        if self.is_distributed:
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            self.ddp_wrapper = torch.nn.parallel.DistributedDataParallel(
                self.ddp_wrapper,
                device_ids=[local_rank] if self.device.type == "cuda" else None,
                broadcast_buffers=False,
            )

        self._trainable_parameter_items = [
            (name, parameter)
            for name, parameter in self.raw_model.named_parameters()
            if parameter.requires_grad
        ]
"""
dino_trainer_content = dino_trainer_content.replace(init_old, init_new)

# Replace remaining `self.model.` with `self.raw_model.`
# but we have to be careful not to replace things indiscriminately.
# Let's just do it directly.
replacements = {
    "getattr(self.model, ": "getattr(self.raw_model, ",
    "self.model.student_backbone": "self.raw_model.student_backbone",
    "self.model.train()": "self.raw_model.train()",
    "self.model.eval()": "self.raw_model.eval()",
    "self.model.teacher_adapter": "self.raw_model.teacher_adapter",
    "self.model.teacher_backbone": "self.raw_model.teacher_backbone",
    "self.model.teacher_head": "self.raw_model.teacher_head",
    "self.model.update_teacher": "self.raw_model.update_teacher",
    "self.model.update_center": "self.raw_model.update_center",
    "self.model.state_dict": "self.raw_model.state_dict",
    "self.model.load_state_dict": "self.raw_model.load_state_dict",
    "self.model.adapt_student": "self.raw_model.adapt_student",
    "self.model.adapt_teacher": "self.raw_model.adapt_teacher",
    "self.model.student_features": "self.raw_model.student_features",
    "self.model.teacher_features": "self.raw_model.teacher_features",
}
for k, v in replacements.items():
    dino_trainer_content = dino_trainer_content.replace(k, v)

# Add is_rank_zero
is_rank_zero_code = """
    @property
    def is_rank_zero(self) -> bool:
        if not self.is_distributed:
            return True
        import torch.distributed as dist
        return dist.get_rank() == 0

    def _autocast_context(self):
"""
dino_trainer_content = dino_trainer_content.replace("    def _autocast_context(self):", is_rank_zero_code)

# 3. train_step replacement
train_step_old = """
        with self._autocast_context():
            student_outputs = self.raw_model.forward_student_views(student_views)       
            teacher_outputs = self.raw_model.forward_teacher_views(teacher_views)       
            student_dense_views = None
            teacher_dense_views = None
            if self.gram_anchor_weight > 0.0:
                student_dense_views = self.raw_model.student_dense_views(student_views[: len(teacher_views)])
                teacher_dense_views = self.raw_model.teacher_dense_views(teacher_views) 
            losses = self.raw_model.dino_v3_loss(
                student_outputs,
                teacher_outputs,
                student_dense_views=student_dense_views,
                teacher_dense_views=teacher_dense_views,
                student_temperature=self.student_temperature,
                gram_anchor_weight=self.gram_anchor_weight,
                gram_anchor_max_tokens=self.gram_anchor_max_tokens,
            )
            loss = losses["loss"]
"""

train_step_new = """
        with self._autocast_context():
            losses = self.ddp_wrapper(
                student_views,
                teacher_views,
                self.student_temperature,
                self.gram_anchor_max_tokens,
            )
            loss = losses["loss"]
"""
dino_trainer_content = dino_trainer_content.replace(train_step_old, train_step_new)

# update_center replacement
center_old = "        self.raw_model.update_center(teacher_outputs, self.center_momentum)"
center_new = """
        with torch.no_grad():
            teacher_outputs = self.raw_model.forward_teacher_views(teacher_views)
            if teacher_outputs:
                batch_center = torch.cat(
                    [output.detach().float() for output in teacher_outputs],
                    dim=0,
                ).mean(dim=0, keepdim=True)
                
                if self.is_distributed:
                    import torch.distributed as dist
                    dist.all_reduce(batch_center)
                    batch_center = batch_center / float(dist.get_world_size())
                    
                self.raw_model.center.mul_(self.center_momentum).add_(
                    batch_center.to(device=self.raw_model.center.device),
                    alpha=1.0 - self.center_momentum,
                )
"""
dino_trainer_content = dino_trainer_content.replace(center_old, center_new)

# 4. evaluate replacement
eval_old = """
                with self._autocast_context():
                    student_outputs = self.raw_model.forward_student_views(student_views)
                    teacher_outputs = self.raw_model.forward_teacher_views(teacher_views)
                    student_dense_views = None
                    teacher_dense_views = None
                    if self.gram_anchor_weight > 0.0:
                        student_dense_views = self.raw_model.student_dense_views(student_views[: len(teacher_views)])
                        teacher_dense_views = self.raw_model.teacher_dense_views(teacher_views)
                    losses_dict = self.raw_model.dino_v3_loss(
                        student_outputs,
                        teacher_outputs,
                        student_dense_views=student_dense_views,
                        teacher_dense_views=teacher_dense_views,
                        student_temperature=self.student_temperature,
                        gram_anchor_weight=self.gram_anchor_weight,
                        gram_anchor_max_tokens=self.gram_anchor_max_tokens,
                    )
                    loss = losses_dict["loss"]
"""
eval_new = """
                with self._autocast_context():
                    losses_dict = self.ddp_wrapper(
                        student_views,
                        teacher_views,
                        self.student_temperature,
                        self.gram_anchor_max_tokens,
                    )
                    loss = losses_dict["loss"]
"""
dino_trainer_content = dino_trainer_content.replace(eval_old, eval_new)

# eval metrics sync
eval_sync_old = """
        mean_loss = float(sum(losses) / len(losses)) if losses else float("nan")    
        mean_cls_loss = float(sum(cls_losses) / len(cls_losses)) if cls_losses else float("nan")
        mean_gram_loss = float(sum(gram_losses) / len(gram_losses)) if gram_losses else float("nan")
        print(
"""
eval_sync_new = """
        if self.is_distributed:
            import torch.distributed as dist
            metrics_tensor = torch.tensor(
                [sum(losses), sum(cls_losses), sum(gram_losses), len(losses)],
                device=self.device, dtype=torch.float64
            )
            dist.all_reduce(metrics_tensor)
            total_loss = float(metrics_tensor[0])
            total_cls_loss = float(metrics_tensor[1])
            total_gram_loss = float(metrics_tensor[2])
            total_batches = int(metrics_tensor[3])
        else:
            total_loss = sum(losses)
            total_cls_loss = sum(cls_losses)
            total_gram_loss = sum(gram_losses)
            total_batches = len(losses)

        mean_loss = float(total_loss / total_batches) if total_batches else float("nan")    
        mean_cls_loss = float(total_cls_loss / total_batches) if total_batches else float("nan")
        mean_gram_loss = float(total_gram_loss / total_batches) if total_batches else float("nan")
        if self.is_rank_zero:
            print(
"""
dino_trainer_content = dino_trainer_content.replace(eval_sync_old, eval_sync_new)
dino_trainer_content = dino_trainer_content.replace(
    '        return {\n            "loss": mean_loss,',
    '        return {\n            "loss": mean_loss,'
).replace(
    'f"val_cls={mean_cls_loss:.6f} val_gram={mean_gram_loss:.6f}"\n        )',
    'f"val_cls={mean_cls_loss:.6f} val_gram={mean_gram_loss:.6f}"\n            )'
)

# train_epoch metrics sync
train_epoch_sync_old = """
        return {
            "loss": float(total_loss / total_batches) if total_batches > 0 else float("nan"),
            "cls_loss": float(total_cls_loss / total_batches) if total_batches > 0 else float("nan"),
            "gram_anchor_loss": float(total_gram_anchor_loss / total_batches) if total_batches > 0 else float("nan"),
            "batches": total_batches,
            "teacher_momentum": float(last_teacher_momentum),
        }
"""
train_epoch_sync_new = """
        if self.is_distributed:
            import torch.distributed as dist
            metrics_tensor = torch.tensor(
                [total_loss, total_cls_loss, total_gram_anchor_loss, total_batches],
                device=self.device, dtype=torch.float64
            )
            dist.all_reduce(metrics_tensor)
            total_loss = float(metrics_tensor[0])
            total_cls_loss = float(metrics_tensor[1])
            total_gram_anchor_loss = float(metrics_tensor[2])
            total_batches = int(metrics_tensor[3])

        return {
            "loss": float(total_loss / total_batches) if total_batches > 0 else float("nan"),
            "cls_loss": float(total_cls_loss / total_batches) if total_batches > 0 else float("nan"),
            "gram_anchor_loss": float(total_gram_anchor_loss / total_batches) if total_batches > 0 else float("nan"),
            "batches": total_batches,
            "teacher_momentum": float(last_teacher_momentum),
        }
"""
dino_trainer_content = dino_trainer_content.replace(train_epoch_sync_old, train_epoch_sync_new)

# log_every print replacement
dino_trainer_content = dino_trainer_content.replace(
    'if (step_index + 1) % self.log_every == 0:\n                print(',
    'if (step_index + 1) % self.log_every == 0 and self.is_rank_zero:\n                print('
)

# 5. fit replacement
# Wrap prints, save_training_checkpoint, _save_visualization, _write_training_metrics
dino_trainer_content = dino_trainer_content.replace(
    '            print(\n                f"train-dino epoch={epoch_index + 1} summary "',
    '            if self.is_rank_zero:\n                print(\n                    f"train-dino epoch={epoch_index + 1} summary "'  # noqa: E501
).replace(
    '                f"duration={epoch_duration:.2f}s"\n            )',
    '                    f"duration={epoch_duration:.2f}s"\n                )'
)

dino_trainer_content = dino_trainer_content.replace(
    '            _write_training_metrics(\n                output_dir,',
    '            if self.is_rank_zero:\n                _write_training_metrics(\n                    output_dir,'
).replace(
    '                best_metric=best_metric,\n            )',
    '                    best_metric=best_metric,\n                )'
)

dino_trainer_content = dino_trainer_content.replace(
    '            curve_path = save_training_curves(history, output_dir, method_name="DINOv3")',
    '            if self.is_rank_zero:\n                curve_path = save_training_curves(history, output_dir, method_name="DINOv3")'  # noqa: E501
).replace(
    '            save_training_checkpoint(checkpoint, output_dir, improved=improved)',
    '            if self.is_rank_zero:\n                save_training_checkpoint(checkpoint, output_dir, improved=improved)'  # noqa: E501
).replace(
    '            visualization_paths = self._save_visualization(output_dir, epoch_index + 1)',
    '            if self.is_rank_zero:\n                visualization_paths = self._save_visualization(output_dir, epoch_index + 1)'  # noqa: E501
).replace(
    '            print(f"[artifacts] Training curve: {curve_path}")\n            for path in visualization_paths:\n                if not path.name.endswith("_latest.png"):\n                    print(f"[artifacts] Model output: {path}")',  # noqa: E501
    '            if self.is_rank_zero:\n                print(f"[artifacts] Training curve: {curve_path}")\n                for path in visualization_paths:\n                    if not path.name.endswith("_latest.png"):\n                        print(f"[artifacts] Model output: {path}")'  # noqa: E501
)

dino_trainer_content = dino_trainer_content.replace(
    '        _write_training_metrics(\n            output_dir,\n            history=history,\n            system_info=system_info,\n            requested_precision=getattr(self.raw_model, "requested_precision", self.requested_precision),\n            resolved_precision=getattr(self.raw_model, "resolved_precision", self.requested_precision),\n            best_metric=best_metric,\n            summary=summary,\n        )',  # noqa: E501
    '        if self.is_rank_zero:\n            _write_training_metrics(\n                output_dir,\n                history=history,\n                system_info=system_info,\n                requested_precision=getattr(self.raw_model, "requested_precision", self.requested_precision),\n                resolved_precision=getattr(self.raw_model, "resolved_precision", self.requested_precision),\n                best_metric=best_metric,\n                summary=summary,\n            )'  # noqa: E501
)

with open(dino_trainer_path, "w", encoding="utf-8") as f:
    f.write(dino_trainer_content)

print("dino_trainer.py updated!")
