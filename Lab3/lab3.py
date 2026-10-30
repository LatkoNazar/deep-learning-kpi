# Лабораторна робота з глибокого навчання (DL)
# Студент гр. КМ-32 Латко Назар Андрійович
# Тема: Обчислювальний експеримент і трансферне навчання

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import subprocess
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from PIL import Image
from sklearn.metrics import confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet18_Weights

CLASSES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASSES)}

IMG_SIZE = 160
CROP_SCALE = (0.7, 1.0)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

TRAIN_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 64
EPOCHS = 8
WEIGHT_DECAY = 1e-4
LR_HEAD = 3e-3
LR_BACKBONE = 3e-4

TOP_LEVEL_MODULES = ("conv1", "bn1", "layer1", "layer2", "layer3", "layer4", "fc")

EXPECTED_SPLIT_CONTROL = {
    "buildings": {"train": (200, 1996761), "validation": (218, 4837323), "test": (219, 4837559)},
    "forest": {"train": (200, 2032625), "validation": (237, 5276962), "test": (237, 5291326)},
    "glacier": {"train": (200, 2183923), "validation": (276, 6135652), "test": (277, 6157250)},
    "mountain": {"train": (200, 1950345), "validation": (262, 5789535), "test": (263, 5849765)},
    "sea": {"train": (200, 1987613), "validation": (255, 5655068), "test": (255, 5668011)},
    "street": {"train": (200, 1841817), "validation": (250, 5541845), "test": (251, 5547533)},
}

EXPECTED_SPLIT_TOTALS = {
    "train": (1200, 11993084),
    "validation": (1498, 33236385),
    "test": (1502, 33351444),
}

EXPECTED_PARAM_COUNTS = {
    "scratch": (11179590, 11179590),
    "frozen": (11179590, 3078),
    "unfreeze": (11179590, 8396806),
}

PROTOCOL_RUNS = [
    ("scratch", 1),
    ("frozen", 1),
    ("frozen", 2),
    ("frozen", 3),
    ("unfreeze", 1),
    ("unfreeze", 2),
    ("unfreeze", 3),
]


def resolve_seg_dir(data_root: Path, seg_name: str) -> Path:
    # перевіряємо стандартну вкладеність архіву seg_train/seg_train та прямий шлях
    nested = data_root / seg_name / seg_name
    direct = data_root / seg_name
    if (nested / CLASSES[0]).is_dir():
        return nested
    if (direct / CLASSES[0]).is_dir():
        return direct
    raise FileNotFoundError(f"Не знайдено теку {seg_name} з підтеками класів у {data_root}")


def select_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def describe_device(device: torch.device) -> str:
    if device.type == "cuda":
        return f"cuda ({torch.cuda.get_device_name(device)})"
    if device.type == "mps":
        return "mps"
    return f"cpu ({platform.processor() or platform.machine()})"


def git_info(workdir: Path) -> tuple[str, bool]:
    try:
        commit_proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=workdir,
            capture_output=True,
            text=True,
            check=True,
        )
        status_proc = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=workdir,
            capture_output=True,
            text=True,
            check=True,
        )
        return commit_proc.stdout.strip(), bool(status_proc.stdout.strip())
    except Exception:
        return "unknown", False


def build_split_csv(data_root: Path, split_path: Path) -> list[dict[str, str]]:
    seg_train_dir = resolve_seg_dir(data_root, "seg_train")
    seg_test_dir = resolve_seg_dir(data_root, "seg_test")

    rows: list[dict[str, str]] = []

    # один генератор для формування train по всіх класах у фіксованому порядку
    rng_train = np.random.default_rng(0)
    for cls_name in CLASSES:
        cls_dir = seg_train_dir / cls_name
        files = sorted(p.name for p in cls_dir.iterdir() if p.is_file() and p.suffix.lower() == ".jpg")
        idx = np.arange(len(files))
        rng_train.shuffle(idx)
        for i in idx[:200]:
            rows.append({"file": files[int(i)], "label": cls_name, "part": "train"})

    # окремий новий генератор для поділу seg_test на validation і test
    rng_eval = np.random.default_rng(0)
    val_rows: list[dict[str, str]] = []
    test_rows: list[dict[str, str]] = []
    for cls_name in CLASSES:
        cls_dir = seg_test_dir / cls_name
        files = sorted(p.name for p in cls_dir.iterdir() if p.is_file() and p.suffix.lower() == ".jpg")
        idx = np.arange(len(files))
        rng_eval.shuffle(idx)
        cut = len(files) // 2
        for i in idx[:cut]:
            val_rows.append({"file": files[int(i)], "label": cls_name, "part": "validation"})
        for i in idx[cut:]:
            test_rows.append({"file": files[int(i)], "label": cls_name, "part": "test"})

    rows.extend(val_rows)
    rows.extend(test_rows)

    split_path.parent.mkdir(parents=True, exist_ok=True)
    with open(split_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "label", "part"])
        writer.writeheader()
        writer.writerows(rows)

    return rows


def read_split_csv(split_path: Path) -> list[dict[str, str]]:
    with open(split_path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def verify_split(rows: list[dict[str, str]]) -> dict[str, object]:
    # перевіряємо відсутність перетинів між частинами поділу
    all_files = [r["file"] for r in rows]
    unique_count = len(set(all_files))
    no_overlap = len(all_files) == unique_count
    if not no_overlap:
        raise ValueError("Виявлено дублювання файлів між частинами поділу split.csv")

    table: dict[str, dict[str, dict[str, object]]] = {}
    totals: dict[str, dict[str, object]] = {}
    all_matched = True

    for cls_name in CLASSES:
        table[cls_name] = {}
        for part in ("train", "validation", "test"):
            part_rows = [r for r in rows if r["label"] == cls_name and r["part"] == part]
            cnt = len(part_rows)
            num_sum = sum(int(Path(r["file"]).stem) for r in part_rows)
            exp_cnt, exp_sum = EXPECTED_SPLIT_CONTROL[cls_name][part]
            matched = (cnt == exp_cnt) and (num_sum == exp_sum)
            all_matched = all_matched and matched
            table[cls_name][part] = {
                "count": cnt,
                "sum": num_sum,
                "expected_count": exp_cnt,
                "expected_sum": exp_sum,
                "matched": matched,
            }

    for part in ("train", "validation", "test"):
        part_rows = [r for r in rows if r["part"] == part]
        cnt = len(part_rows)
        num_sum = sum(int(Path(r["file"]).stem) for r in part_rows)
        exp_cnt, exp_sum = EXPECTED_SPLIT_TOTALS[part]
        matched = (cnt == exp_cnt) and (num_sum == exp_sum)
        all_matched = all_matched and matched
        totals[part] = {
            "count": cnt,
            "sum": num_sum,
            "expected_count": exp_cnt,
            "expected_sum": exp_sum,
            "matched": matched,
        }

    if not all_matched:
        raise ValueError("Контрольна таблиця split.csv не збігається з умовою завдання")

    return {
        "no_overlap": no_overlap,
        "total_unique_files": unique_count,
        "all_cells_matched": all_matched,
        "by_class": table,
        "totals": totals,
    }


def make_transforms(part: str) -> transforms.Compose:
    if part == "train":
        return transforms.Compose([
            transforms.RandomResizedCrop(IMG_SIZE, scale=CROP_SCALE),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class IntelSplitDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]], part: str, data_root: Path):
        if part not in ("train", "validation", "test"):
            raise ValueError(f"Невідома частина набору: {part}")
        self.part = part
        self.items = [r for r in rows if r["part"] == part]
        seg_name = "seg_train" if part == "train" else "seg_test"
        self.base_dir = resolve_seg_dir(data_root, seg_name)
        self.transform = make_transforms(part)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.items[index]
        img_path = self.base_dir / row["label"] / row["file"]
        with Image.open(img_path) as img:
            rgb = img.convert("RGB")
        tensor = self.transform(rgb)
        label = CLASS_TO_IDX[row["label"]]
        return tensor, label


def eval_module_names(regime: str) -> tuple[str, ...]:
    if regime == "scratch":
        return ()
    if regime == "frozen":
        return ("conv1", "bn1", "layer1", "layer2", "layer3", "layer4")
    if regime == "unfreeze":
        return ("conv1", "bn1", "layer1", "layer2", "layer3")
    raise ValueError(f"Невідомий режим: {regime}")


def set_train_mode(model: nn.Module, regime: str) -> None:
    # вмикаємо режим навчання та переводимо заморожені модулі у eval()
    model.train()
    for mod_name in eval_module_names(regime):
        getattr(model, mod_name).eval()


def build_model_and_optimizer(
    regime: str,
    seed: int,
    device: torch.device,
) -> tuple[nn.Module, torch.optim.Optimizer]:
    # фіксований порядок кроків 1-5 з умови завдання
    torch.manual_seed(seed)
    np.random.seed(seed)

    weights = None if regime == "scratch" else ResNet18_Weights.IMAGENET1K_V1
    model = torchvision.models.resnet18(weights=weights)
    model.fc = nn.Linear(512, len(CLASSES))

    if regime == "scratch":
        for p in model.parameters():
            p.requires_grad = True
    elif regime == "frozen":
        for p in model.parameters():
            p.requires_grad = False
        for p in model.fc.parameters():
            p.requires_grad = True
    elif regime == "unfreeze":
        for p in model.parameters():
            p.requires_grad = False
        for p in model.layer4.parameters():
            p.requires_grad = True
        for p in model.fc.parameters():
            p.requires_grad = True
    else:
        raise ValueError(f"Невідомий режим: {regime}")

    model = model.to(device=device, dtype=torch.float32)

    if regime == "scratch":
        param_groups = [{"params": list(model.parameters()), "lr": LR_HEAD}]
    elif regime == "frozen":
        param_groups = [{"params": list(model.fc.parameters()), "lr": LR_HEAD}]
    else:
        param_groups = [
            {"params": list(model.fc.parameters()), "lr": LR_HEAD},
            {"params": list(model.layer4.parameters()), "lr": LR_BACKBONE},
        ]

    optimizer = torch.optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)
    return model, optimizer


def make_train_loader(train_ds: Dataset) -> DataLoader:
    return DataLoader(
        train_ds,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        drop_last=False,
        num_workers=0,
    )


def make_eval_loader(eval_ds: Dataset) -> DataLoader:
    return DataLoader(
        eval_ds,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )


def save_batch_preview(train_ds: Dataset, out_path: Path) -> None:
    torch.manual_seed(1)
    np.random.seed(1)
    loader = make_train_loader(train_ds)
    xb, yb = next(iter(loader))

    mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.array(IMAGENET_STD, dtype=np.float32).reshape(1, 3, 1, 1)
    imgs = xb[:16].numpy() * std + mean
    imgs = np.clip(np.transpose(imgs, (0, 2, 3, 1)), 0.0, 1.0)
    labels = yb[:16].numpy()

    fig, axes = plt.subplots(4, 4, figsize=(8.5, 8.8))
    for i, ax in enumerate(axes.flat):
        ax.imshow(imgs[i])
        ax.set_title(CLASSES[int(labels[i])], fontsize=10)
        ax.axis("off")
    fig.suptitle("Перші 16 зображень навчального пакета після перетворень", fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def check_initial_losses(train_ds: Dataset, device: torch.device) -> dict[str, float]:
    criterion = nn.CrossEntropyLoss()
    result = {"ln6": float(math.log(len(CLASSES)))}
    for regime in ("scratch", "frozen"):
        model, _ = build_model_and_optimizer(regime, seed=1, device=device)
        loader = make_train_loader(train_ds)
        set_train_mode(model, regime)
        xb, yb = next(iter(loader))
        with torch.no_grad():
            logits = model(xb.to(device=device, dtype=torch.float32))
            loss0 = criterion(logits, yb.to(device=device, dtype=torch.long)).item()
        result[regime] = float(loss0)
    return result


def check_regime_parameters(device: torch.device) -> dict[str, object]:
    info: dict[str, object] = {}
    for regime in ("scratch", "frozen", "unfreeze"):
        model, optimizer = build_model_and_optimizer(regime, seed=1, device=device)
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        exp_total, exp_trainable = EXPECTED_PARAM_COUNTS[regime]

        has_frozen_in_opt = any(
            not p.requires_grad for group in optimizer.param_groups for p in group["params"]
        )
        groups = [
            {
                "num_params": int(sum(p.numel() for p in group["params"])),
                "lr": float(group["lr"]),
                "weight_decay": float(group["weight_decay"]),
            }
            for group in optimizer.param_groups
        ]
        matched = (total == exp_total) and (trainable == exp_trainable) and (not has_frozen_in_opt)
        if not matched:
            raise AssertionError(f"Помилка перевірки параметрів у режимі {regime}")

        info[regime] = {
            "total_params": int(total),
            "trainable_params": int(trainable),
            "optimizer_groups": groups,
            "frozen_in_optimizer": bool(has_frozen_in_opt),
            "matched": matched,
        }
    return info


def inspect_module_changes(
    model: nn.Module,
    before_state: dict[str, torch.Tensor],
    after_state: dict[str, torch.Tensor],
    expected_frozen_modules: set[str],
) -> list[dict[str, object]]:
    rows = []
    for mod_name in TOP_LEVEL_MODULES:
        module = getattr(model, mod_name)
        param_keys = [f"{mod_name}.{name}" for name, _ in module.named_parameters()]
        params_changed = any(not torch.equal(before_state[k], after_state[k]) for k in param_keys)

        bn_prefixes = []
        for sub_name, sub_mod in module.named_modules():
            if isinstance(sub_mod, nn.BatchNorm2d):
                bn_prefixes.append(f"{mod_name}.{sub_name}" if sub_name else mod_name)

        if not bn_prefixes:
            bn_status = "немає BN"
        else:
            changed_bn = any(
                (not torch.equal(before_state[f"{pfx}.running_mean"], after_state[f"{pfx}.running_mean"]))
                or (not torch.equal(before_state[f"{pfx}.running_var"], after_state[f"{pfx}.running_var"]))
                for pfx in bn_prefixes
            )
            bn_status = "так" if changed_bn else "ні"

        is_frozen = mod_name in expected_frozen_modules
        exp_params = not is_frozen
        if not bn_prefixes:
            exp_bn = "немає BN"
        else:
            exp_bn = "ні" if is_frozen else "так"

        matched = (params_changed == exp_params) and (bn_status == exp_bn)
        rows.append({
            "module": mod_name,
            "params_changed": "так" if params_changed else "ні",
            "bn_stats_changed": bn_status,
            "matched_expected": "так" if matched else "ні",
        })
    return rows


def run_five_step_checks(train_ds: Dataset, device: torch.device) -> dict[str, object]:
    criterion = nn.CrossEntropyLoss()
    results: dict[str, object] = {}

    for regime in ("frozen", "unfreeze"):
        model, optimizer = build_model_and_optimizer(regime, seed=1, device=device)
        loader = make_train_loader(train_ds)
        before_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        set_train_mode(model, regime)
        for step_idx, (xb, yb) in enumerate(loader):
            if step_idx >= 5:
                break
            optimizer.zero_grad()
            loss = criterion(
                model(xb.to(device=device, dtype=torch.float32)),
                yb.to(device=device, dtype=torch.long),
            )
            loss.backward()
            optimizer.step()

        after_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        rows = inspect_module_changes(model, before_state, after_state, set(eval_module_names(regime)))
        if not all(r["matched_expected"] == "так" for r in rows):
            raise AssertionError(f"Перевірка 5 кроків не збіглася з очікуванням у режимі {regime}")
        results[regime] = rows

    # дослід із навмисною помилкою: model.train() замість set_train_mode для frozen
    bug_model, bug_opt = build_model_and_optimizer("frozen", seed=1, device=device)
    bug_loader = make_train_loader(train_ds)
    before_bug = {k: v.detach().cpu().clone() for k, v in bug_model.state_dict().items()}

    bug_model.train()
    for step_idx, (xb, yb) in enumerate(bug_loader):
        if step_idx >= 5:
            break
        bug_opt.zero_grad()
        loss = criterion(
            bug_model(xb.to(device=device, dtype=torch.float32)),
            yb.to(device=device, dtype=torch.long),
        )
        loss.backward()
        bug_opt.step()

    after_bug = {k: v.detach().cpu().clone() for k, v in bug_model.state_dict().items()}
    bug_rows = inspect_module_changes(bug_model, before_bug, after_bug, set(eval_module_names("frozen")))

    mean_diff = (
        after_bug["layer1.0.bn1.running_mean"] - before_bug["layer1.0.bn1.running_mean"]
    ).abs().max().item()
    var_diff = (
        after_bug["layer1.0.bn1.running_var"] - before_bug["layer1.0.bn1.running_var"]
    ).abs().max().item()

    results["frozen_bug_experiment"] = {
        "modules": bug_rows,
        "layer1_0_bn1_max_abs_running_mean_diff": float(mean_diff),
        "layer1_0_bn1_max_abs_running_var_diff": float(var_diff),
    }
    return results


@torch.no_grad()
def predict_logits(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_logits = []
    all_targets = []
    for xb, yb in loader:
        logits = model(xb.to(device=device, dtype=torch.float32))
        all_logits.append(logits.detach().cpu().numpy().astype(np.float32))
        all_targets.append(yb.numpy().astype(np.int64))
    return np.concatenate(all_logits, axis=0), np.concatenate(all_targets, axis=0)


def execute_single_run(
    regime: str,
    seed: int,
    train_ds: Dataset,
    val_ds: Dataset,
    device: torch.device,
    split_sha256: str,
    commit_hash: str,
    is_dirty: bool,
    runs_dir: Path,
) -> tuple[dict[str, object], dict[str, torch.Tensor]]:
    model, optimizer = build_model_and_optimizer(regime, seed=seed, device=device)
    train_loader = make_train_loader(train_ds)
    val_loader = make_eval_loader(val_ds)
    criterion = nn.CrossEntropyLoss()

    train_losses: list[float] = []
    val_f1s: list[float] = []
    n_train = len(train_ds)
    val_logits_last = np.empty((0, len(CLASSES)), dtype=np.float32)

    t_start = time.perf_counter()
    for epoch in range(EPOCHS):
        set_train_mode(model, regime)
        epoch_loss_sum = 0.0
        for xb, yb in train_loader:
            xb_dev = xb.to(device=device, dtype=torch.float32)
            yb_dev = yb.to(device=device, dtype=torch.long)
            optimizer.zero_grad()
            logits = model(xb_dev)
            loss = criterion(logits, yb_dev)
            loss.backward()
            optimizer.step()
            epoch_loss_sum += float(loss.detach().item()) * len(yb)

        epoch_loss = epoch_loss_sum / n_train
        val_logits, val_targets = predict_logits(model, val_loader, device=device)
        val_preds = np.argmax(val_logits, axis=1)
        val_f1 = float(f1_score(val_targets, val_preds, average="macro"))

        train_losses.append(float(epoch_loss))
        val_f1s.append(val_f1)
        val_logits_last = val_logits
        print(
            f"  [{regime} seed={seed}] epoch {epoch + 1}/{EPOCHS} "
            f"train_loss={epoch_loss:.6f} val_macro_f1={val_f1:.6f}"
        )

    elapsed = float(time.perf_counter() - t_start)

    run_dir = runs_dir / f"{regime}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    np.save(run_dir / "val_logits.npy", val_logits_last)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    journal = {
        "regime": regime,
        "seed": seed,
        "transforms": {
            "img_size": IMG_SIZE,
            "train_random_resized_crop_scale": list(CROP_SCALE),
            "train_random_resized_crop_ratio": [0.75, 4.0 / 3.0],
            "train_random_horizontal_flip_p": 0.5,
            "eval_resize": [IMG_SIZE, IMG_SIZE],
            "interpolation": "bilinear",
            "normalize_mean": list(IMAGENET_MEAN),
            "normalize_std": list(IMAGENET_STD),
            "test_time_augmentation": False,
        },
        "model": {
            "architecture": "resnet18",
            "weights": None if regime == "scratch" else "ResNet18_Weights.IMAGENET1K_V1",
            "classes": CLASSES,
            "num_classes": len(CLASSES),
            "total_params": int(total_params),
            "trainable_params": int(trainable_params),
            "eval_modules_during_train": list(eval_module_names(regime)),
            "bn_momentum": 0.1,
            "bn_eps": 1e-5,
            "dtype": "float32",
        },
        "training": {
            "optimizer": "AdamW",
            "optimizer_groups": [
                {
                    "num_params": int(sum(p.numel() for p in g["params"])),
                    "lr": float(g["lr"]),
                    "weight_decay": float(g["weight_decay"]),
                    "betas": list(g["betas"]),
                    "eps": float(g["eps"]),
                    "amsgrad": bool(g["amsgrad"]),
                }
                for g in optimizer.param_groups
            ],
            "loss_fn": "CrossEntropyLoss(reduction='mean', label_smoothing=0.0, weight=None)",
            "epochs": EPOCHS,
            "train_batch_size": TRAIN_BATCH_SIZE,
            "eval_batch_size": EVAL_BATCH_SIZE,
            "steps_per_epoch": math.ceil(n_train / TRAIN_BATCH_SIZE),
            "total_optimizer_steps": EPOCHS * math.ceil(n_train / TRAIN_BATCH_SIZE),
            "shuffle_train": True,
            "drop_last": False,
            "num_workers": 0,
            "lr_scheduler": None,
            "grad_clip": None,
            "mixed_precision": False,
            "early_stopping": False,
            "checkpoint_selection_by_epoch": False,
        },
        "git_commit": commit_hash,
        "git_uncommitted_changes": is_dirty,
        "split_sha256": split_sha256,
        "environment": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "torchvision": str(torchvision.__version__),
            "device": describe_device(device),
        },
        "train_loss_by_epoch": train_losses,
        "val_macro_f1_by_epoch": val_f1s,
        "final_train_loss": train_losses[-1],
        "final_val_macro_f1": val_f1s[-1],
        "training_time_sec": elapsed,
    }

    write_json(run_dir / "run.json", journal)
    cpu_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return journal, cpu_state


def select_best_from_journals(journals: list[dict[str, object]]) -> dict[str, object]:
    by_regime: dict[str, list[dict[str, object]]] = {"scratch": [], "frozen": [], "unfreeze": []}
    for j in journals:
        by_regime[str(j["regime"])].append(j)

    regime_means = {
        reg: float(np.mean([float(x["final_val_macro_f1"]) for x in items]))
        for reg, items in by_regime.items()
    }
    best_regime = max(("scratch", "frozen", "unfreeze"), key=lambda r: regime_means[r])
    best_run = max(by_regime[best_regime], key=lambda x: float(x["final_val_macro_f1"]))
    return {
        "regime_means": regime_means,
        "selected_regime": best_regime,
        "selected_seed": int(best_run["seed"]),
        "selected_val_macro_f1": float(best_run["final_val_macro_f1"]),
    }


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def train_command(data_root: Path, out_dir: Path, device_arg: str = "auto") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = out_dir / "figures"
    runs_dir = out_dir / "runs"

    device = select_device(device_arg)
    print(f"Пристрій для навчання: {describe_device(device)}")

    # попередньо кешуємо ваги ImageNet до запуску вимірювань часу
    _ = ResNet18_Weights.IMAGENET1K_V1.get_state_dict(progress=True)

    # крок 1: побудова та перевірка поділу split.csv (зображення test не відкриваються)
    split_path = out_dir / "split.csv"
    rows = build_split_csv(data_root, split_path)
    split_check = verify_split(rows)
    split_sha256 = hashlib.sha256(split_path.read_bytes()).hexdigest()
    print(f"Поділ збережено у {split_path} (SHA-256: {split_sha256[:16]}...), усі 18 клітинок збіглися.")

    train_ds = IntelSplitDataset(rows, part="train", data_root=data_root)
    val_ds = IntelSplitDataset(rows, part="validation", data_root=data_root)

    # перевірки до навчання (пункт 1)
    save_batch_preview(train_ds, figures_dir / "batch_preview.png")
    l0_check = check_initial_losses(train_ds, device=device)
    print(
        f"Початкова втрата L0: scratch={l0_check['scratch']:.6f}, "
        f"frozen={l0_check['frozen']:.6f}, ln(6)={l0_check['ln6']:.6f}"
    )

    # перевірки режимів навчання та дослід із BN (пункт 2)
    param_check = check_regime_parameters(device=device)
    five_step_check = run_five_step_checks(train_ds, device=device)
    print("Перевірки параметрів та 5 кроків навчання пройдено успішно.")

    checks_payload = {
        "split_sha256": split_sha256,
        "split_verification": split_check,
        "initial_loss_L0": l0_check,
        "parameter_verification": param_check,
        "five_step_verification": five_step_check,
    }
    write_json(out_dir / "pretrain_checks.json", checks_payload)

    # сім запусків основного протоколу (пункт 3)
    commit_hash, is_dirty = git_info(out_dir)
    journals: list[dict[str, object]] = []
    states_by_run: dict[tuple[str, int], dict[str, torch.Tensor]] = {}

    for run_idx, (regime, seed) in enumerate(PROTOCOL_RUNS, start=1):
        print(f"Запуск {run_idx}/7: regime={regime}, seed={seed}")
        journal, state_dict = execute_single_run(
            regime=regime,
            seed=seed,
            train_ds=train_ds,
            val_ds=val_ds,
            device=device,
            split_sha256=split_sha256,
            commit_hash=commit_hash,
            is_dirty=is_dirty,
            runs_dir=runs_dir,
        )
        journals.append(journal)
        states_by_run[(regime, seed)] = state_dict

    # вибір найкращої конфігурації виключно за validation (пункт 5)
    selection = select_best_from_journals(journals)
    best_regime = str(selection["selected_regime"])
    best_seed = int(selection["selected_seed"])
    best_state = states_by_run[(best_regime, best_seed)]

    model_path = out_dir / "lab3_model.pt"
    torch.save(
        {
            "state_dict": best_state,
            "regime": best_regime,
            "seed": best_seed,
            "classes": CLASSES,
            "img_size": IMG_SIZE,
        },
        model_path,
    )
    print(
        f"Обрано модель: regime={best_regime}, seed={best_seed}, "
        f"val_macro_f1={selection['selected_val_macro_f1']:.6f} -> {model_path}"
    )


def save_learning_curves(journals: list[dict[str, object]], out_path: Path) -> None:
    epochs = np.arange(1, EPOCHS + 1)
    fig, (ax_loss, ax_f1) = plt.subplots(1, 2, figsize=(12.5, 4.8))

    styles = {
        "scratch": ("#d95f02", "-"),
        "frozen": ("#1b9e77", "--"),
        "unfreeze": ("#7570b3", "-."),
    }
    markers = {1: "o", 2: "s", 3: "^"}

    for j in journals:
        reg = str(j["regime"])
        seed = int(j["seed"])
        color, ls = styles[reg]
        mk = markers[seed]
        label = f"{reg} (seed {seed})"
        ax_loss.plot(
            epochs,
            j["train_loss_by_epoch"],
            label=label,
            color=color,
            linestyle=ls,
            marker=mk,
            markersize=4.5,
            linewidth=1.5,
            alpha=0.9,
        )
        ax_f1.plot(
            epochs,
            j["val_macro_f1_by_epoch"],
            label=label,
            color=color,
            linestyle=ls,
            marker=mk,
            markersize=4.5,
            linewidth=1.5,
            alpha=0.9,
        )

    ax_loss.set_title("Втрата на train за епохами")
    ax_loss.set_xlabel("Епоха")
    ax_loss.set_ylabel("Cross-entropy loss")
    ax_loss.set_xticks(epochs)
    ax_loss.grid(alpha=0.25)

    ax_f1.set_title("Macro-F1 на validation за епохами")
    ax_f1.set_xlabel("Епоха")
    ax_f1.set_ylabel("Validation macro-F1")
    ax_f1.set_xticks(epochs)
    ax_f1.grid(alpha=0.25)
    ax_f1.legend(loc="lower right", fontsize=8.5)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_confusion_matrix_figure(cm: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 5.8))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(np.arange(len(CLASSES)))
    ax.set_yticks(np.arange(len(CLASSES)))
    ax.set_xticklabels(CLASSES, rotation=30, ha="right")
    ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Передбачений клас")
    ax.set_ylabel("Справжній клас")
    ax.set_title("Матриця плутанини на test (6×6)")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                str(int(cm[i, j])),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=9.5,
            )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def select_characteristic_errors(
    test_rows: list[dict[str, str]],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray,
    cm: np.ndarray,
) -> list[dict[str, object]]:
    # позадіагональні пари за спаданням частоти, далі за індексом справжнього і передбаченого класу
    pairs = []
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            if i != j and cm[i, j] > 0:
                pairs.append((int(cm[i, j]), i, j))
    pairs.sort(key=lambda t: (-t[0], t[1], t[2]))

    files_by_pair: dict[tuple[int, int], list[tuple[str, int]]] = {}
    for idx, row in enumerate(test_rows):
        ti = int(y_true[idx])
        pi = int(y_pred[idx])
        if ti != pi:
            files_by_pair.setdefault((ti, pi), []).append((row["file"], idx))

    for key in files_by_pair:
        files_by_pair[key].sort(key=lambda item: item[0])

    selected: list[dict[str, object]] = []
    rank = 0
    while len(selected) < 5 and pairs:
        added_in_round = False
        for count, ti, pi in pairs:
            candidates = files_by_pair[(ti, pi)]
            if rank < len(candidates):
                fname, sample_idx = candidates[rank]
                selected.append({
                    "file": fname,
                    "true_class": CLASSES[ti],
                    "pred_class": CLASSES[pi],
                    "pred_probability": float(probs[sample_idx, pi]),
                    "pair_error_count": int(count),
                })
                added_in_round = True
                if len(selected) == 5:
                    break
        if not added_in_round:
            break
        rank += 1

    return selected


def save_characteristic_errors_figure(
    errors: list[dict[str, object]],
    seg_test_dir: Path,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, len(errors), figsize=(14.5, 4.0))
    if len(errors) == 1:
        axes = [axes]

    for ax, err in zip(axes, errors):
        img_path = seg_test_dir / str(err["true_class"]) / str(err["file"])
        with Image.open(img_path) as img:
            rgb = img.convert("RGB")
            arr = np.asarray(rgb)
        ax.imshow(arr)
        ax.set_title(
            f"{err['file']}\ntrue: {err['true_class']}\n"
            f"pred: {err['pred_class']} ({err['pred_probability']:.3f})\n"
            f"пар у test: {err['pair_error_count']}",
            fontsize=8.5,
        )
        ax.axis("off")

    fig.suptitle("П'ять характерних помилок моделі на test", fontsize=11.5, y=0.97)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def analyze_command(data_root: Path, out_dir: Path) -> dict[str, object]:
    split_path = out_dir / "split.csv"
    runs_dir = out_dir / "runs"
    figures_dir = out_dir / "figures"
    model_path = out_dir / "lab3_model.pt"

    if not split_path.exists():
        raise FileNotFoundError(f"Не знайдено {split_path}")
    current_sha256 = hashlib.sha256(split_path.read_bytes()).hexdigest()

    rows = read_split_csv(split_path)
    split_check = verify_split(rows)
    val_labels = np.array([CLASS_TO_IDX[r["label"]] for r in rows if r["part"] == "validation"], dtype=np.int64)

    # завантажуємо рівно 7 журналів протоколу та звіряємо хеш split.csv і val_logits.npy
    journals: list[dict[str, object]] = []
    runs_table: list[dict[str, object]] = []

    for run_num, (regime, seed) in enumerate(PROTOCOL_RUNS, start=1):
        run_dir = runs_dir / f"{regime}_seed{seed}"
        journal_path = run_dir / "run.json"
        logits_path = run_dir / "val_logits.npy"
        if not journal_path.exists() or not logits_path.exists():
            raise FileNotFoundError(f"Відсутній журнал або val_logits.npy для {regime}_seed{seed}")

        with open(journal_path, "r", encoding="utf-8") as f:
            journal = json.load(f)

        if journal.get("split_sha256") != current_sha256:
            raise RuntimeError(
                f"Хеш split.csv у {journal_path} ({journal.get('split_sha256')}) "
                f"не збігається з поточним split.csv ({current_sha256})"
            )

        val_logits = np.load(logits_path)
        if val_logits.shape != (1498, len(CLASSES)):
            raise ValueError(f"Некоректна форма val_logits.npy у {run_dir}: {val_logits.shape}")
        recomputed_f1 = float(f1_score(val_labels, np.argmax(val_logits, axis=1), average="macro"))
        if abs(recomputed_f1 - float(journal["final_val_macro_f1"])) > 1e-12:
            raise ValueError(f"Розбіжність macro-F1 між run.json та val_logits.npy у {run_dir}")

        journals.append(journal)
        runs_table.append({
            "run_id": run_num,
            "regime": regime,
            "seed": seed,
            "train_loss_epoch8": float(journal["final_train_loss"]),
            "val_macro_f1_epoch8": float(journal["final_val_macro_f1"]),
            "training_time_sec": float(journal["training_time_sec"]),
        })

    save_learning_curves(journals, figures_dir / "learning_curves.png")

    # пункт 4: розклад ефектів
    f_scratch = float(journals[0]["final_val_macro_f1"])
    frozen_scores = [float(j["final_val_macro_f1"]) for j in journals if j["regime"] == "frozen"]
    unfreeze_scores = [float(j["final_val_macro_f1"]) for j in journals if j["regime"] == "unfreeze"]

    f_bar_frozen = float(np.mean(frozen_scores))
    r_frozen = float(np.max(frozen_scores) - np.min(frozen_scores))
    f_bar_unfreeze = float(np.mean(unfreeze_scores))
    r_unfreeze = float(np.max(unfreeze_scores) - np.min(unfreeze_scores))

    r_noise = float(max(r_frozen, r_unfreeze))
    delta_pre = float(f_bar_frozen - f_scratch)
    delta_unf = float(f_bar_unfreeze - f_bar_frozen)
    total_transfer_gain = float(f_bar_unfreeze - f_scratch)

    verdict_pre = (
        "не відрізняється від розкиду"
        if abs(delta_pre) <= r_noise
        else "ефект вартий перевірки, але не доведений"
    )
    verdict_unf = (
        "не відрізняється від розкиду"
        if abs(delta_unf) <= r_noise
        else "ефект вартий перевірки, але не доведений"
    )

    # пункт 5: перевірка вибору моделі та одноразова оцінка test на CPU
    selection = select_best_from_journals(journals)
    if not model_path.exists():
        raise FileNotFoundError(f"Не знайдено збережену модель {model_path}")

    ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
    if (
        ckpt.get("regime") != selection["selected_regime"]
        or int(ckpt.get("seed")) != int(selection["selected_seed"])
        or list(ckpt.get("classes")) != CLASSES
        or int(ckpt.get("img_size")) != IMG_SIZE
    ):
        raise RuntimeError("Файл lab3_model.pt не відповідає правилу вибору за збереженими журналами")

    cpu_device = torch.device("cpu")
    model = torchvision.models.resnet18(weights=None)
    model.fc = nn.Linear(512, len(CLASSES))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device=cpu_device, dtype=torch.float32)
    model.eval()

    test_ds = IntelSplitDataset(rows, part="test", data_root=data_root)
    test_loader = make_eval_loader(test_ds)
    test_logits, test_targets = predict_logits(model, test_loader, device=cpu_device)

    shifted = test_logits - np.max(test_logits, axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    test_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    test_preds = np.argmax(test_logits, axis=1)

    test_macro_f1 = float(f1_score(test_targets, test_preds, average="macro"))
    test_accuracy = float(np.mean(test_preds == test_targets))
    n_test = len(test_targets)
    test_se_accuracy = float(math.sqrt(test_accuracy * (1.0 - test_accuracy) / n_test))

    cm = confusion_matrix(test_targets, test_preds, labels=list(range(len(CLASSES))))
    save_confusion_matrix_figure(cm, figures_dir / "confusion_matrix.png")

    test_rows = [r for r in rows if r["part"] == "test"]
    characteristic_errors = select_characteristic_errors(
        test_rows=test_rows,
        y_true=test_targets,
        y_pred=test_preds,
        probs=test_probs,
        cm=cm,
    )
    seg_test_dir = resolve_seg_dir(data_root, "seg_test")
    save_characteristic_errors_figure(
        characteristic_errors,
        seg_test_dir=seg_test_dir,
        out_path=figures_dir / "top5_errors.png",
    )

    pretrain_checks = None
    checks_path = out_dir / "pretrain_checks.json"
    if checks_path.exists():
        with open(checks_path, "r", encoding="utf-8") as f:
            pretrain_checks = json.load(f)

    summary = {
        "split_sha256": current_sha256,
        "split_totals": split_check["totals"],
        "pretrain_checks": pretrain_checks,
        "runs_table": runs_table,
        "effects": {
            "F_scratch_1": f_scratch,
            "F_bar_frozen": f_bar_frozen,
            "R_frozen": r_frozen,
            "F_bar_unfreeze": f_bar_unfreeze,
            "R_unfreeze": r_unfreeze,
            "R": r_noise,
            "delta_pre": delta_pre,
            "verdict_pre": verdict_pre,
            "delta_unf": delta_unf,
            "verdict_unf": verdict_unf,
            "total_transfer_gain": total_transfer_gain,
            "selection_determined_by_noise": bool(abs(delta_unf) <= r_noise),
        },
        "selection": selection,
        "test": {
            "n_test": int(n_test),
            "macro_f1": test_macro_f1,
            "selected_val_macro_f1": float(selection["selected_val_macro_f1"]),
            "test_minus_val_macro_f1": float(test_macro_f1 - float(selection["selected_val_macro_f1"])),
            "accuracy": test_accuracy,
            "se_accuracy": test_se_accuracy,
            "classes": CLASSES,
            "confusion_matrix": cm.tolist(),
            "characteristic_errors": characteristic_errors,
        },
    }

    write_json(out_dir / "analysis_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    return summary


def make_parser() -> argparse.ArgumentParser:
    default_out = Path(__file__).resolve().parent
    default_data = default_out.parent / "intel-image-classification"

    parser = argparse.ArgumentParser(description="Лабораторна робота 3")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_train = subparsers.add_parser("train", help="поділ, перевірки, сім запусків і вибір моделі")
    p_train.add_argument("--data", type=Path, default=default_data, help="шлях до Intel Image Classification")
    p_train.add_argument("--out", type=Path, default=default_out, help="тека для результатів ЛР3")
    p_train.add_argument("--device", type=str, default="auto", help="пристрій навчання (auto/cpu/cuda/mps)")

    p_analyze = subparsers.add_parser("analyze", help="перевірка журналів, розклад ефектів та оцінка на test")
    p_analyze.add_argument("--data", type=Path, default=default_data, help="шлях до Intel Image Classification")
    p_analyze.add_argument("--out", type=Path, default=default_out, help="тека з результатами ЛР3")

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = make_parser()
    args = parser.parse_args(argv)
    if args.command == "train":
        train_command(args.data, args.out, device_arg=args.device)
    elif args.command == "analyze":
        analyze_command(args.data, args.out)


if __name__ == "__main__":
    main()
