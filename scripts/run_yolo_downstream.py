"""
run_yolo_downstream.py
======================
Runs ALL experiments in the YAML config sequentially, then prints a
comparison table.  Uses the same Rich progress bar as spark_runner.py.
The Rich bar is inlined here to avoid any sys.path shadowing issues.

Usage:
    conda run -n ag-foundation python scripts/run_yolo_downstream.py
    conda run -n ag-foundation python scripts/run_yolo_downstream.py --config configs/det_plantseg_yolo.yaml
"""

import yaml
import argparse
import sys
import os
import time
import shutil

from ultralytics import YOLO

# ── Inline Rich progress bar (identical style to ag_foundation/progress.py) ──
from rich.console import Console
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    SpinnerColumn,
    MofNCompleteColumn,
)


def _make_progress(label: str) -> Progress:
    """Create the same Rich Progress bar used across the codebase."""
    stream = sys.stdout
    force_term = None
    if os.name == "nt" and not stream.isatty():
        try:
            stream = open("CONOUT$", "w", encoding="utf-8")
            force_term = True
        except OSError:
            pass

    term_width = shutil.get_terminal_size(fallback=(120, 24)).columns
    console = Console(file=stream, force_terminal=force_term, width=term_width)

    return Progress(
        SpinnerColumn(spinner_name="dots", style="bold bright_green"),
        TextColumn("[bold bright_cyan]{task.description}"),
        BarColumn(bar_width=40, complete_style="bright_green",
                  finished_style="bold bright_green", pulse_style="bright_white"),
        "[progress.percentage]{task.percentage:>3.1f}%",
        "•",
        MofNCompleteColumn(),
        "•",
        TextColumn("[dim]Elapsed:[/dim]"),
        TimeElapsedColumn(),
        "•",
        TextColumn("[dim]ETA:[/dim]"),
        TimeRemainingColumn(),
        "•",
        TextColumn("[bold bright_yellow]{task.fields[detail]}"),
        console=console,
        transient=False,
        refresh_per_second=10,
    )


# ──────────────────────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _sep(char="─", width=90):
    print(char * width, flush=True)


# ──────────────────────────────────────────────────────────────────────────────
#  Single experiment runner
# ──────────────────────────────────────────────────────────────────────────────

def run_experiment(cfg: dict, exp: dict) -> dict:
    """Train one model and return its best metrics dict."""

    name         = exp["name"]
    model_path   = exp["model"]
    label        = exp.get("label", name)
    total_epochs = cfg["epochs"]

    _sep("=")
    print(f"  Experiment : {label}", flush=True)
    print(f"  Model      : {model_path}", flush=True)
    print(f"  Output     : {cfg['project']}/{name}", flush=True)
    _sep("=")
    print("", flush=True)

    model = YOLO(model_path)

    # ── Rich progress bar ─────────────────────────────────────────────────
    progress = _make_progress(label)
    task_id  = None
    progress.start()

    state = {
        "step":        0,
        "total":       0,
        "epoch":       0,
        "epoch_start": 0.0,
        "best_mAP50":  0.0,
        "best_epoch":  0,
        "final":       {},
    }

    # ── Callbacks ─────────────────────────────────────────────────────────
    def on_train_epoch_start(trainer):
        nonlocal task_id
        state["step"]        = 0
        state["total"]       = len(trainer.train_loader)
        state["epoch"]       = trainer.epoch + 1
        state["epoch_start"] = time.time()

        desc = f"Train Epoch [{state['epoch']}/{total_epochs}]"
        if task_id is None:
            task_id = progress.add_task(desc, total=state["total"], detail="starting…")
        else:
            progress.reset(task_id, total=state["total"], description=desc, detail="starting…")

    def on_train_batch_end(trainer):
        state["step"] += 1
        step  = state["step"]

        items = trainer.loss_items
        if items is not None:
            loss_labels = ["box", "cls", "dfl", "ang"]
            detail = "  ".join(
                f"{loss_labels[i]}={items[i].item():.4f}"
                for i in range(min(len(items), len(loss_labels)))
            )
        else:
            detail = "loss=n/a"

        if task_id is not None:
            progress.update(task_id, completed=state["step"], detail=detail)

    def on_fit_epoch_end(trainer):
        epoch   = state["epoch"]
        elapsed = time.time() - state["epoch_start"]
        metrics = trainer.metrics if hasattr(trainer, "metrics") else {}
        items   = trainer.loss_items
        lr      = trainer.lr if hasattr(trainer, "lr") else {}
        lr0     = list(lr.values())[0] if lr else float("nan")

        loss_labels = ["box", "cls", "dfl", "ang"]
        loss_str = "  ".join(
            f"{loss_labels[i]}={items[i].item():.4f}"
            for i in range(min(len(items), len(loss_labels)))
        ) if items is not None else "n/a"

        mAP50   = metrics.get("metrics/mAP50(B)",    0.0)
        mAP9595 = metrics.get("metrics/mAP50-95(B)", 0.0)
        P       = metrics.get("metrics/precision(B)", 0.0)
        R       = metrics.get("metrics/recall(B)",    0.0)
        fitness = metrics.get("fitness",              0.0)

        is_best = mAP50 > state["best_mAP50"]
        if is_best:
            state["best_mAP50"] = mAP50
            state["best_epoch"] = epoch
            state["final"]      = dict(metrics)

        best_tag = "  ★ NEW BEST" if is_best else ""

        print(f"\nEpoch {epoch:>3}/{total_epochs} Complete  ({elapsed:.1f}s){best_tag}", flush=True)
        print(f"  Train Loss  →  {loss_str}  |  LR: {lr0:.6f}", flush=True)
        print(
            f"  Val Metrics →  mAP50: {mAP50:.4f}  mAP50-95: {mAP9595:.4f}"
            f"  P: {P:.4f}  R: {R:.4f}  Fitness: {fitness:.4f}",
            flush=True,
        )
        print(
            f"  Best so far →  mAP50: {state['best_mAP50']:.4f} @ epoch {state['best_epoch']}\n",
            flush=True,
        )

    def on_train_end(trainer):
        if task_id is not None:
            total = state["total"]
            progress.update(task_id, completed=total, detail="[bold green]Completed")
        progress.stop()
        _sep("=")
        print(f"  [{label}] Training complete!", flush=True)
        print(f"  Best mAP50 : {state['best_mAP50']:.4f} @ epoch {state['best_epoch']}", flush=True)
        print(f"  Saved to   : {trainer.save_dir}", flush=True)
        _sep("=")
        print("", flush=True)

    model.add_callback("on_train_epoch_start", on_train_epoch_start)
    model.add_callback("on_train_batch_end",   on_train_batch_end)
    model.add_callback("on_fit_epoch_end",     on_fit_epoch_end)
    model.add_callback("on_train_end",         on_train_end)

    model.train(
        data             = cfg["data"],
        epochs           = total_epochs,
        batch            = cfg.get("batch",          16),
        imgsz            = cfg.get("imgsz",         640),
        workers          = cfg.get("workers",          8),
        lr0              = cfg.get("lr0",          0.001),
        lrf              = cfg.get("lrf",           0.01),
        weight_decay     = cfg.get("weight_decay", 0.0005),
        warmup_epochs    = cfg.get("warmup_epochs",  3.0),
        momentum         = cfg.get("momentum",     0.937),
        optimizer        = cfg.get("optimizer",  "AdamW"),
        patience         = cfg.get("patience",       15),
        cos_lr           = cfg.get("cos_lr",        True),
        label_smoothing  = cfg.get("label_smoothing", 0.1),
        dropout          = cfg.get("dropout",        0.0),
        hsv_h            = cfg.get("hsv_h",        0.015),
        hsv_s            = cfg.get("hsv_s",          0.7),
        hsv_v            = cfg.get("hsv_v",          0.4),
        degrees          = cfg.get("degrees",        0.0),
        translate        = cfg.get("translate",      0.1),
        scale            = cfg.get("scale",          0.5),
        flipud           = cfg.get("flipud",         0.0),
        fliplr           = cfg.get("fliplr",         0.5),
        mosaic           = cfg.get("mosaic",         1.0),
        mixup            = cfg.get("mixup",          0.0),
        task             = cfg.get("task",          "obb"),
        project          = cfg.get("project", r"E:\AG_Dataset\AG-Foundational-Model\runs"),
        name             = name,
        verbose          = False,
    )

    return {"label": label, "name": name, **state["final"]}


# ──────────────────────────────────────────────────────────────────────────────
#  Comparison table
# ──────────────────────────────────────────────────────────────────────────────

def print_comparison(results: list[dict]):
    _sep("=")
    print("  EXPERIMENT COMPARISON  –  PlantSeg OBB (YOLO11-L)", flush=True)
    _sep("=")

    col_w = 38
    print(
        f"  {'Model':<{col_w}}"
        f"{'mAP50':>10}"
        f"{'mAP50-95':>12}"
        f"{'Precision':>12}"
        f"{'Recall':>10}"
        f"{'Fitness':>10}",
        flush=True,
    )
    _sep()

    for r in results:
        print(
            f"  {r['label']:<{col_w}}"
            f"{r.get('metrics/mAP50(B)', 0.0):>10.4f}"
            f"{r.get('metrics/mAP50-95(B)', 0.0):>12.4f}"
            f"{r.get('metrics/precision(B)', 0.0):>12.4f}"
            f"{r.get('metrics/recall(B)', 0.0):>10.4f}"
            f"{r.get('fitness', 0.0):>10.4f}",
            flush=True,
        )

    _sep("=")
    print("", flush=True)


# ──────────────────────────────────────────────────────────────────────────────
#  Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main(config_path: str):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    experiments = cfg.get("experiments", [])
    if not experiments:
        print("[ERROR] No 'experiments:' list found in config.", flush=True)
        sys.exit(1)

    total_exp = len(experiments)
    _sep("=")
    print(f"  Starting {total_exp} experiment(s)  –  running sequentially", flush=True)
    for i, exp in enumerate(experiments, 1):
        print(f"    {i}. {exp.get('label', exp['name'])}  →  {exp['model']}", flush=True)
    _sep("=")
    print("", flush=True)

    all_results = []
    for i, exp in enumerate(experiments, 1):
        print(f"\n{'#' * 90}", flush=True)
        print(f"#  [{i}/{total_exp}]  {exp.get('label', exp['name'])}", flush=True)
        print(f"{'#' * 90}\n", flush=True)
        result = run_experiment(cfg, exp)
        all_results.append(result)

    print_comparison(all_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO OBB comparison experiments")
    parser.add_argument(
        "--config",
        type=str,
        default=r"E:\AG_Dataset\AG-Foundational-Model\configs\det_plantseg_yolo.yaml",
    )
    args = parser.parse_args()
    main(args.config)
