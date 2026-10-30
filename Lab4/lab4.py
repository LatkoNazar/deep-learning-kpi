# Лабораторна робота з глибокого навчання (DL)
# Студент гр. КМ-32 Латко Назар Андрійович
# Тема: FP32 проти INT8 — статична квантизація після навчання в ONNX Runtime

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper
import onnxruntime as ort
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)
from PIL import Image
from sklearn.metrics import f1_score
import torch
from torch import nn
import torchvision
from torchvision import transforms

try:
    from onnxruntime.quantization import quant_pre_process
except ImportError:
    from onnxruntime.quantization.shape_inference import quant_pre_process

EXPECTED_CLASSES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
CLASS_TO_IDX = {name: i for i, name in enumerate(EXPECTED_CLASSES)}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

EXPECTED_TRAIN_SPLIT = (1200, 11993084)
EXPECTED_TEST_SPLIT = (1502, 33351444)

EXPECTED_CALIB_COUNT = 200
EXPECTED_CALIB_SUM = 1921470
EXPECTED_CALIB_BY_CLASS = {
    "buildings": 40,
    "forest": 35,
    "glacier": 36,
    "mountain": 33,
    "sea": 29,
    "street": 27,
}

EVAL_BATCH_SIZE = 64
LATENCY_ROUNDS = 3
WARMUP_CALLS = 20
TIMED_CALLS = 100


def get_cpu_model_name() -> str:
    sys_name = platform.system()
    if sys_name == "Windows":
        try:
            import winreg

            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            ) as key:
                val, _ = winreg.QueryValueEx(key, "ProcessorNameString")
                if val:
                    return str(val).strip()
        except Exception:
            pass
    elif sys_name == "Darwin":
        try:
            proc = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                check=True,
            )
            if proc.stdout.strip():
                return proc.stdout.strip()
        except Exception:
            pass
    elif sys_name == "Linux":
        try:
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="ignore").splitlines():
                if "model name" in line:
                    return line.split(":", 1)[1].strip()
        except Exception:
            pass
    return platform.processor() or platform.machine()


def resolve_seg_dir(data_root: Path, seg_name: str) -> Path:
    nested = data_root / seg_name / seg_name
    direct = data_root / seg_name
    if (nested / EXPECTED_CLASSES[0]).is_dir():
        return nested
    if (direct / EXPECTED_CLASSES[0]).is_dir():
        return direct
    raise FileNotFoundError(f"Не знайдено теку {seg_name} у {data_root}")


def load_pytorch_model(model_path: Path) -> tuple[nn.Module, dict[str, object]]:
    ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
    classes = list(ckpt["classes"])
    img_size = int(ckpt["img_size"])
    if classes != EXPECTED_CLASSES:
        raise ValueError(f"Порядок класів у моделі {classes} не збігається з {EXPECTED_CLASSES}")

    model = torchvision.models.resnet18(weights=None)
    model.fc = nn.Linear(512, len(classes))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device="cpu", dtype=torch.float32)
    model.eval()
    return model, {
        "regime": str(ckpt["regime"]),
        "seed": int(ckpt["seed"]),
        "classes": classes,
        "img_size": img_size,
    }


def read_and_verify_split(split_path: Path) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, object]]:
    with open(split_path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    train_rows = [r for r in rows if r["part"] == "train"]
    test_rows = [r for r in rows if r["part"] == "test"]

    train_cnt = len(train_rows)
    train_sum = sum(int(Path(r["file"]).stem) for r in train_rows)
    test_cnt = len(test_rows)
    test_sum = sum(int(Path(r["file"]).stem) for r in test_rows)

    train_ok = (train_cnt, train_sum) == EXPECTED_TRAIN_SPLIT
    test_ok = (test_cnt, test_sum) == EXPECTED_TEST_SPLIT
    if not (train_ok and test_ok):
        raise ValueError(
            f"Контроль split.csv не збігається: "
            f"train=({train_cnt}, {train_sum}), test=({test_cnt}, {test_sum})"
        )

    return train_rows, test_rows, {
        "train": {"count": train_cnt, "sum_numeric_names": train_sum, "matched": train_ok},
        "test": {"count": test_cnt, "sum_numeric_names": test_sum, "matched": test_ok},
    }


def make_eval_transform(img_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def load_images_array(
    rows: list[dict[str, str]],
    base_dir: Path,
    img_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    transform = make_eval_transform(img_size)
    tensors = np.empty((len(rows), 3, img_size, img_size), dtype=np.float32)
    labels = np.empty(len(rows), dtype=np.int64)
    for idx, row in enumerate(rows):
        img_path = base_dir / row["label"] / row["file"]
        with Image.open(img_path) as img:
            rgb = img.convert("RGB")
        tensors[idx] = transform(rgb).numpy()
        labels[idx] = CLASS_TO_IDX[row["label"]]
    return tensors, labels


def build_calibration_subset(train_rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], dict[str, object]]:
    # сортуємо 1200 рядків train за іменем файлу як рядки та беремо перші 200 із перестановки
    sorted_train = sorted(train_rows, key=lambda r: r["file"])
    idx = np.random.default_rng(0).permutation(len(sorted_train))[:EXPECTED_CALIB_COUNT]
    calib_rows = [sorted_train[int(i)] for i in idx]

    cnt = len(calib_rows)
    num_sum = sum(int(Path(r["file"]).stem) for r in calib_rows)
    by_class = {
        cls_name: sum(1 for r in calib_rows if r["label"] == cls_name)
        for cls_name in EXPECTED_CLASSES
    }

    matched = (
        cnt == EXPECTED_CALIB_COUNT
        and num_sum == EXPECTED_CALIB_SUM
        and by_class == EXPECTED_CALIB_BY_CLASS
    )
    if not matched:
        raise ValueError(
            f"Контроль калібрувальної вибірки не збігається: count={cnt}, sum={num_sum}, by_class={by_class}"
        )

    return calib_rows, {
        "count": cnt,
        "sum_numeric_names": num_sum,
        "by_class": by_class,
        "matched": matched,
    }


class SingleImageCalibrationReader(CalibrationDataReader):
    def __init__(self, calib_images: np.ndarray):
        self._samples = [
            {"input": np.ascontiguousarray(calib_images[i : i + 1], dtype=np.float32)}
            for i in range(len(calib_images))
        ]
        self._iter = iter(self._samples)

    def get_next(self) -> dict[str, np.ndarray] | None:
        return next(self._iter, None)

    def rewind(self) -> None:
        self._iter = iter(self._samples)


def create_cpu_session(onnx_path: Path) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    return ort.InferenceSession(
        str(onnx_path),
        sess_options=opts,
        providers=["CPUExecutionProvider"],
    )


def export_fp32_onnx(model: nn.Module, img_size: int, out_path: Path) -> float:
    model.eval()
    dummy = torch.zeros(1, 3, img_size, img_size, dtype=torch.float32)
    t0 = time.perf_counter()
    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        dynamo=False,
        opset_version=17,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
    )
    return float(time.perf_counter() - t0)


@torch.no_grad()
def run_pytorch_batches(model: nn.Module, images: np.ndarray, batch_size: int = EVAL_BATCH_SIZE) -> np.ndarray:
    model.eval()
    outputs = []
    for start in range(0, len(images), batch_size):
        batch = torch.from_numpy(images[start : start + batch_size])
        outputs.append(model(batch).numpy().astype(np.float32))
    return np.concatenate(outputs, axis=0)


def run_onnx_batches(
    session: ort.InferenceSession,
    images: np.ndarray,
    batch_size: int = EVAL_BATCH_SIZE,
) -> np.ndarray:
    outputs = []
    for start in range(0, len(images), batch_size):
        batch = np.ascontiguousarray(images[start : start + batch_size], dtype=np.float32)
        (logits,) = session.run(["logits"], {"input": batch})
        outputs.append(logits.astype(np.float32))
    return np.concatenate(outputs, axis=0)


def inspect_onnx_graph_and_channel_ranges(fp32_path: Path) -> dict[str, object]:
    model_proto = onnx.load(str(fp32_path))
    graph = model_proto.graph
    initializers = {init.name: numpy_helper.to_array(init) for init in graph.initializer}

    conv_nodes = [n for n in graph.node if n.op_type == "Conv"]
    bn_nodes = [n for n in graph.node if n.op_type == "BatchNormalization"]

    if len(conv_nodes) != 20 or len(bn_nodes) != 0:
        raise ValueError(
            f"Неочікувана кількість вузлів графа: Conv={len(conv_nodes)}, BatchNormalization={len(bn_nodes)}"
        )

    conv_stats = []
    for idx, node in enumerate(conv_nodes):
        node_name = node.name or f"Conv_{idx}"
        weight_name = node.input[1]
        weights = initializers[weight_name]
        alpha_c = np.max(np.abs(weights), axis=(1, 2, 3))
        max_alpha = float(np.max(alpha_c))
        med_alpha = float(np.median(alpha_c))
        ratio_r = float(max_alpha / med_alpha)
        conv_stats.append({
            "node_name": node_name,
            "weight_name": weight_name,
            "shape": list(weights.shape),
            "max_alpha_c": max_alpha,
            "median_alpha_c": med_alpha,
            "r": ratio_r,
            "bits_cost_log2_r": float(math.log2(ratio_r)),
        })

    sorted_by_r = sorted(conv_stats, key=lambda x: x["r"], reverse=True)
    median_r = float(np.median([x["r"] for x in conv_stats]))
    median_bits_cost = float(math.log2(median_r))

    return {
        "conv_count": len(conv_nodes),
        "batch_norm_count": len(bn_nodes),
        "top3_conv_by_r": sorted_by_r[:3],
        "median_r": median_r,
        "median_channel_bits_cost": median_bits_cost,
        "all_conv_layers": conv_stats,
    }


def evaluate_variant(
    y_true: np.ndarray,
    preds_variant: np.ndarray,
    preds_fp32: np.ndarray,
) -> dict[str, object]:
    n = len(y_true)
    macro_f1 = float(f1_score(y_true, preds_variant, average="macro"))
    acc = float(np.mean(preds_variant == y_true))
    agreement = float(np.mean(preds_variant == preds_fp32))

    b = int(np.sum((preds_variant == y_true) & (preds_fp32 != y_true)))
    c = int(np.sum((preds_variant != y_true) & (preds_fp32 == y_true)))
    delta_acc = float((b - c) / n)
    se_delta = float(math.sqrt(b + c) / n)
    threshold_2sqrt = float(2.0 * math.sqrt(b + c))
    diff_bc = abs(b - c)

    if diff_bc <= threshold_2sqrt:
        verdict = "різниця якості не відрізняється від шуму"
    elif b > c:
        verdict = "різниця відрізняється від шуму (INT8 точніша за FP32)"
    else:
        verdict = "різниця відрізняється від шуму (INT8 поступається FP32)"

    return {
        "n_test": int(n),
        "macro_f1": macro_f1,
        "accuracy": acc,
        "agreement_with_fp32": agreement,
        "b_int8_right_fp32_wrong": b,
        "c_int8_wrong_fp32_right": c,
        "delta_acc": delta_acc,
        "se_delta_acc": se_delta,
        "abs_b_minus_c": int(diff_bc),
        "noise_bound_2sqrt_bc": threshold_2sqrt,
        "verdict": verdict,
    }


def measure_latency_protocol(
    sess_fp32: ort.InferenceSession,
    sess_int8: ort.InferenceSession,
    first_image: np.ndarray,
) -> dict[str, object]:
    feed = {"input": np.ascontiguousarray(first_image[0:1], dtype=np.float32)}
    if feed["input"].shape != (1, 3, first_image.shape[2], first_image.shape[3]):
        raise ValueError(f"Некоректна форма входу для вимірювання затримки: {feed['input'].shape}")

    fp32_rounds: list[list[float]] = []
    int8_rounds: list[list[float]] = []

    for _ in range(LATENCY_ROUNDS):
        # прогрів та 100 вимірів для FP32
        for _ in range(WARMUP_CALLS):
            sess_fp32.run(None, feed)
        cur_fp32 = [0.0] * TIMED_CALLS
        for i in range(TIMED_CALLS):
            t0 = time.perf_counter()
            sess_fp32.run(None, feed)
            t1 = time.perf_counter()
            cur_fp32[i] = (t1 - t0) * 1000.0
        fp32_rounds.append(cur_fp32)

        # прогрів та 100 вимірів для INT8
        for _ in range(WARMUP_CALLS):
            sess_int8.run(None, feed)
        cur_int8 = [0.0] * TIMED_CALLS
        for i in range(TIMED_CALLS):
            t0 = time.perf_counter()
            sess_int8.run(None, feed)
            t1 = time.perf_counter()
            cur_int8[i] = (t1 - t0) * 1000.0
        int8_rounds.append(cur_int8)

    fp32_all = [x for rnd in fp32_rounds for x in rnd]
    int8_all = [x for rnd in int8_rounds for x in rnd]

    fp32_p50_rounds = [float(np.percentile(rnd, 50)) for rnd in fp32_rounds]
    fp32_p95_rounds = [float(np.percentile(rnd, 95)) for rnd in fp32_rounds]
    int8_p50_rounds = [float(np.percentile(rnd, 50)) for rnd in int8_rounds]
    int8_p95_rounds = [float(np.percentile(rnd, 95)) for rnd in int8_rounds]

    fp32_p50 = float(np.percentile(fp32_all, 50))
    fp32_p95 = float(np.percentile(fp32_all, 95))
    int8_p50 = float(np.percentile(int8_all, 50))
    int8_p95 = float(np.percentile(int8_all, 95))

    fp32_interval = [min(fp32_p50_rounds), max(fp32_p50_rounds)]
    int8_interval = [min(int8_p50_rounds), max(int8_p50_rounds)]
    disjoint = (fp32_interval[1] < int8_interval[0]) or (int8_interval[1] < fp32_interval[0])

    return {
        "fp32": {
            "p50_ms": fp32_p50,
            "p95_ms": fp32_p95,
            "p50_by_round_ms": fp32_p50_rounds,
            "p95_by_round_ms": fp32_p95_rounds,
            "round_p50_min_max_ms": fp32_interval,
            "all_300_measurements_ms": fp32_all,
        },
        "int8": {
            "p50_ms": int8_p50,
            "p95_ms": int8_p95,
            "p50_by_round_ms": int8_p50_rounds,
            "p95_by_round_ms": int8_p95_rounds,
            "round_p50_min_max_ms": int8_interval,
            "all_300_measurements_ms": int8_all,
        },
        "speedup_p50": float(fp32_p50 / int8_p50),
        "p95_ratio": float(fp32_p95 / int8_p95),
        "latency_change_measured": bool(disjoint),
    }


def run_pipeline(model_path: Path, split_path: Path, data_root: Path, out_dir: Path) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)

    # завантаження моделі ЛР3 та перевірка поділу split.csv
    pt_model, model_meta = load_pytorch_model(model_path)
    img_size = int(model_meta["img_size"])
    train_rows, test_rows, split_check = read_and_verify_split(split_path)

    seg_train_dir = resolve_seg_dir(data_root, "seg_train")
    seg_test_dir = resolve_seg_dir(data_root, "seg_test")

    calib_rows, calib_check = build_calibration_subset(train_rows)
    calib_images, _ = load_images_array(calib_rows, seg_train_dir, img_size=img_size)
    test_images, test_labels = load_images_array(test_rows, seg_test_dir, img_size=img_size)

    fp32_onnx_path = out_dir / "lab4_fp32.onnx"
    prep_onnx_path = out_dir / "lab4_prep.onnx"
    int8_onnx_path = out_dir / "lab4_int8.onnx"
    int8_pt_onnx_path = out_dir / "lab4_int8_per_tensor.onnx"

    # пункт 1: експорт у ONNX та перевірка відповідності
    export_time_sec = export_fp32_onnx(pt_model, img_size=img_size, out_path=fp32_onnx_path)
    graph_and_ranges = inspect_onnx_graph_and_channel_ranges(fp32_onnx_path)

    sess_fp32 = create_cpu_session(fp32_onnx_path)
    logits_pt = run_pytorch_batches(pt_model, test_images)
    logits_fp32 = run_onnx_batches(sess_fp32, test_images)

    max_abs_logit_diff = float(np.max(np.abs(logits_fp32 - logits_pt)))
    if max_abs_logit_diff >= 1e-3:
        raise ValueError(f"Занадто велика розбіжність логітів ONNX і PyTorch: {max_abs_logit_diff}")

    preds_pt = np.argmax(logits_pt, axis=1)
    preds_fp32 = np.argmax(logits_fp32, axis=1)
    export_class_agreement = float(np.mean(preds_fp32 == preds_pt))
    f1_pt = float(f1_score(test_labels, preds_pt, average="macro"))
    f1_fp32 = float(f1_score(test_labels, preds_fp32, average="macro"))

    # пункт 2: підготовка графа, калібрування та статична поканальна квантизація
    t_q0 = time.perf_counter()
    quant_pre_process(str(fp32_onnx_path), str(prep_onnx_path), skip_symbolic_shape=True)
    t_prep = time.perf_counter() - t_q0

    t_q1 = time.perf_counter()
    quantize_static(
        model_input=str(prep_onnx_path),
        model_output=str(int8_onnx_path),
        calibration_data_reader=SingleImageCalibrationReader(calib_images),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        calibrate_method=CalibrationMethod.MinMax,
    )
    t_qstatic = time.perf_counter() - t_q1
    quantization_time_sec = float(t_prep + t_qstatic)

    # пункт 3: оцінка якості FP32 та INT8 (per-channel)
    sess_int8 = create_cpu_session(int8_onnx_path)
    logits_int8 = run_onnx_batches(sess_int8, test_images)
    preds_int8 = np.argmax(logits_int8, axis=1)

    fp32_quality = evaluate_variant(test_labels, preds_fp32, preds_fp32)
    int8_quality = evaluate_variant(test_labels, preds_int8, preds_fp32)
    se_single_model = float(
        math.sqrt(fp32_quality["accuracy"] * (1.0 - fp32_quality["accuracy"]) / len(test_labels))
    )

    # пункт 4: прогноз до запуску та запуск із потензорними вагами (per_channel=False)
    prior_prediction = {
        "agreement_will_drop": True,
        "macro_f1_change_beyond_noise": False,
        "rationale": (
            "Медіанне відношення діапазонів каналів r становить близько 3-4 (~1.8 біта втрати точності "
            "на медіанний канал), а максимальне менше 8 (<3 бітів), на відміну від depthwise-згорток "
            "MobileNetV3 (r ~ 90). Тому через грубіший крок квантування окремі примежові об'єкти змінять "
            "клас і згода передбачень з FP32 дещо знизиться порівняно з per-channel, але різниця за "
            "точністю та macro-F1 залишиться в межах випадкового шуму (|b - c| <= 2*sqrt(b + c))."
        ),
    }

    t_pt0 = time.perf_counter()
    quantize_static(
        model_input=str(prep_onnx_path),
        model_output=str(int8_pt_onnx_path),
        calibration_data_reader=SingleImageCalibrationReader(calib_images),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=False,
        calibrate_method=CalibrationMethod.MinMax,
    )
    per_tensor_quant_time_sec = float(time.perf_counter() - t_pt0)

    sess_int8_pt = create_cpu_session(int8_pt_onnx_path)
    logits_int8_pt = run_onnx_batches(sess_int8_pt, test_images)
    preds_int8_pt = np.argmax(logits_int8_pt, axis=1)
    int8_per_tensor_quality = evaluate_variant(test_labels, preds_int8_pt, preds_fp32)

    # пункт 5: вимірювання затримки за протоколом (B=1, 1 потік, 3 раунди по 20+100 викликів)
    latency = measure_latency_protocol(sess_fp32, sess_int8, test_images)

    fp32_bytes = int(fp32_onnx_path.stat().st_size)
    int8_bytes = int(int8_onnx_path.stat().st_size)
    int8_pt_bytes = int(int8_pt_onnx_path.stat().st_size)

    # видаляємо проміжний службовий файл підготовки графа
    if prep_onnx_path.exists():
        prep_onnx_path.unlink()

    results = {
        "model_metadata": model_meta,
        "split_verification": split_check,
        "calibration_verification": calib_check,
        "timings_sec": {
            "onnx_export_sec": export_time_sec,
            "quant_pre_process_sec": float(t_prep),
            "quantize_static_per_channel_sec": float(t_qstatic),
            "total_quantization_sec": quantization_time_sec,
            "quantize_static_per_tensor_sec": per_tensor_quant_time_sec,
        },
        "section1_export_verification": {
            "max_abs_logit_diff": max_abs_logit_diff,
            "class_agreement": export_class_agreement,
            "macro_f1_pytorch": f1_pt,
            "macro_f1_onnx_fp32": f1_fp32,
            "conv_nodes": graph_and_ranges["conv_count"],
            "batch_norm_nodes": graph_and_ranges["batch_norm_count"],
        },
        "section3_quality": {
            "fp32": fp32_quality,
            "int8_per_channel": int8_quality,
            "single_model_se_accuracy": se_single_model,
            "se_single_to_paired_ratio": (
                float(se_single_model / int8_quality["se_delta_acc"])
                if int8_quality["se_delta_acc"] > 0
                else None
            ),
        },
        "section4_per_tensor_experiment": {
            "top3_conv_by_r": graph_and_ranges["top3_conv_by_r"],
            "median_r": graph_and_ranges["median_r"],
            "median_channel_bits_cost": graph_and_ranges["median_channel_bits_cost"],
            "all_conv_layers": graph_and_ranges["all_conv_layers"],
            "prior_prediction": prior_prediction,
            "int8_per_tensor_quality": int8_per_tensor_quality,
        },
        "section5_latency_and_size": {
            "fp32_size_bytes": fp32_bytes,
            "fp32_size_mb": float(fp32_bytes / 1_000_000.0),
            "int8_size_bytes": int8_bytes,
            "int8_size_mb": float(int8_bytes / 1_000_000.0),
            "int8_per_tensor_size_mb": float(int8_pt_bytes / 1_000_000.0),
            "size_reduction_ratio": float(fp32_bytes / int8_bytes),
            "latency": latency,
        },
        "environment": {
            "cpu_model": get_cpu_model_name(),
            "os": f"{platform.system()} {platform.release()} ({platform.version()})",
            "architecture": platform.machine(),
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "torchvision": str(torchvision.__version__),
            "onnx": str(onnx.__version__),
            "onnxruntime": str(ort.__version__),
            "providers": ["CPUExecutionProvider"],
            "intra_op_num_threads": 1,
            "inter_op_num_threads": 1,
            "latency_batch_size": 1,
            "quality_batch_size": EVAL_BATCH_SIZE,
        },
    }

    results_path = out_dir / "lab4_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")

    # компактний вивід основних чисел у консоль
    summary_for_stdout = {k: v for k, v in results.items() if k != "section5_latency_and_size"}
    summary_for_stdout["section5_latency_summary"] = {
        "fp32_size_mb": results["section5_latency_and_size"]["fp32_size_mb"],
        "int8_size_mb": results["section5_latency_and_size"]["int8_size_mb"],
        "size_reduction_ratio": results["section5_latency_and_size"]["size_reduction_ratio"],
        "fp32_p50_ms": latency["fp32"]["p50_ms"],
        "fp32_p95_ms": latency["fp32"]["p95_ms"],
        "fp32_p50_by_round_ms": latency["fp32"]["p50_by_round_ms"],
        "int8_p50_ms": latency["int8"]["p50_ms"],
        "int8_p95_ms": latency["int8"]["p95_ms"],
        "int8_p50_by_round_ms": latency["int8"]["p50_by_round_ms"],
        "speedup_p50": latency["speedup_p50"],
        "latency_change_measured": latency["latency_change_measured"],
    }
    print(json.dumps(summary_for_stdout, ensure_ascii=False, indent=2, allow_nan=False))
    return results


def resolve_default_lab3_file(lab4_dir: Path, filename: str) -> Path:
    for folder_name in ("Lab3", "lab3"):
        candidate = lab4_dir.parent / folder_name / filename
        if candidate.exists():
            return candidate
    return lab4_dir.parent / "Lab3" / filename


def make_parser() -> argparse.ArgumentParser:
    default_out = Path(__file__).resolve().parent
    default_model = resolve_default_lab3_file(default_out, "lab3_model.pt")
    default_split = resolve_default_lab3_file(default_out, "split.csv")
    default_data = default_out.parent / "intel-image-classification"

    parser = argparse.ArgumentParser(description="Лабораторна робота 4: FP32 проти INT8")
    parser.add_argument("--model", type=Path, default=default_model, help="шлях до lab3_model.pt")
    parser.add_argument("--split", type=Path, default=default_split, help="шлях до split.csv із ЛР3")
    parser.add_argument("--data", type=Path, default=default_data, help="шлях до Intel Image Classification")
    parser.add_argument("--out", type=Path, default=default_out, help="тека для збереження lab4_results.json")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = make_parser()
    args = parser.parse_args(argv)
    run_pipeline(
        model_path=args.model,
        split_path=args.split,
        data_root=args.data,
        out_dir=args.out,
    )


if __name__ == "__main__":
    main()
