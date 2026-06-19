import os
import re

dino_runner_path = "src/ag_foundation/training/dino_runner.py"
with open(dino_runner_path, "r", encoding="utf-8") as f:
    dino_runner_content = f.read()

# 1. Update run_train_dino signature/body
run_old = """
def run_train_dino(args: argparse.Namespace, *, command_argv: list[str] | None = None):
    set_global_seed(args.seed)
    resume_checkpoint = _resolve_resume_checkpoint(args)
"""
run_new = """
def run_train_dino(args: argparse.Namespace, *, command_argv: list[str] | None = None):
    import os
    import torch.distributed as dist

    is_distributed = "LOCAL_RANK" in os.environ
    if is_distributed and args.device in {None, "auto"}:
        args.device = f"cuda:{os.environ['LOCAL_RANK']}"

    set_global_seed(args.seed)
    resume_checkpoint = _resolve_resume_checkpoint(args)
"""
dino_runner_content = dino_runner_content.replace(run_old, run_new)

# 2. Update manifest creation to only happen on rank 0
manifest_old = """
    manifest_path = write_run_manifest(args.output_dir, manifest)
    print(f"[metadata] Saved run manifest to {manifest_path}")
"""
manifest_new = """
    is_rank_zero = not is_distributed or dist.get_rank() == 0
    if is_rank_zero:
        manifest_path = write_run_manifest(args.output_dir, manifest)
        print(f"[metadata] Saved run manifest to {manifest_path}")
"""
dino_runner_content = dino_runner_content.replace(manifest_old, manifest_new)

# 3. Update main() to initialize and teardown DDP
main_old = """
def main(argv: list[str] | None = None) -> None:
    args = parse_train_dino_args(argv)
    summary = run_train_dino(args, command_argv=list(argv or []))
    print(summary)
"""
main_new = """
def main(argv: list[str] | None = None) -> None:
    import os
    import torch
    is_distributed = "LOCAL_RANK" in os.environ
    if is_distributed:
        import torch.distributed as dist
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
            torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    args = parse_train_dino_args(argv)
    summary = run_train_dino(args, command_argv=list(argv or []))

    is_rank_zero = not is_distributed or dist.get_rank() == 0
    if is_rank_zero:
        print(summary)

    if is_distributed and dist.is_initialized():
        dist.destroy_process_group()
"""
dino_runner_content = dino_runner_content.replace(main_old, main_new)

with open(dino_runner_path, "w", encoding="utf-8") as f:
    f.write(dino_runner_content)

print("dino_runner.py updated!")
