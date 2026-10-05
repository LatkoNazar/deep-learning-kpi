# Лабораторна робота з глибокого навчання (DL)
# Студент гр. КМ-32 Латко Назар Андрійович
# Тема: Поріг рішення під асиметричну ціну помилок

from pathlib import Path
import argparse
import json
import platform
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.metrics import auc, precision_recall_curve, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn

# константи та параметри за умовою
SEED = 0
EPOCHS = 12
BATCH_SIZE = 1024
LEARNING_RATE = 0.001
HIDDEN_UNITS = 32
FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount"]


class Split:
    def __init__(self, x, y, amount, row_ids):
        self.x = x
        self.y = y
        self.amount = amount
        self.row_ids = row_ids


def make_splits(csv_path):
    # читаємо дані з файлу, перевіряємо потрібні колонки
    frame = pd.read_csv(csv_path)
    required = set(FEATURES + ["Class"])
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"У CSV бракує стовпців: {', '.join(missing)}")

    labels = frame["Class"].to_numpy(dtype=np.int64)
    raw_features = frame[FEATURES].to_numpy(dtype=np.float64)

    # спочатку виділяємо train та тимчасову частину
    all_rows = np.arange(len(frame))
    train_rows, held_rows = train_test_split(
        all_rows, test_size=0.4, stratify=labels, random_state=SEED
    )
    # тимчасову частину ділимо порівну на validation та test
    val_rows, test_rows = train_test_split(
        held_rows, test_size=0.5, stratify=labels[held_rows], random_state=SEED
    )

    # сортуємо індекси, щоб зберегти початковий порядок рядків
    train_rows.sort()
    val_rows.sort()
    test_rows.sort()

    # scaler навчаємо тільки на навчальній вибірці
    scaler = StandardScaler().fit(raw_features[train_rows])

    def create_split(rows):
        std_x = scaler.transform(raw_features[rows]).astype(np.float32)
        return Split(
            x=std_x,
            y=labels[rows],
            amount=raw_features[rows, -1],
            row_ids=rows,
        )

    return create_split(train_rows), create_split(val_rows), create_split(test_rows), scaler


def build_model():
    # фіксуємо зерно перед створенням шарів
    torch.manual_seed(SEED)
    model = nn.Sequential(
        nn.Linear(len(FEATURES), HIDDEN_UNITS),
        nn.ReLU(),
        nn.Linear(HIDDEN_UNITS, 1),
    )
    return model.to(device="cpu", dtype=torch.float32)


def train_model(train_split):
    model = build_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.BCEWithLogitsLoss()

    train_x = torch.from_numpy(train_split.x)
    train_y = torch.from_numpy(train_split.y.astype(np.float32)).unsqueeze(1)

    # окремий генератор для випадкового перемішування train на кожній епосі
    rng = np.random.default_rng(SEED)
    losses = []
    model.train()

    n_samples = len(train_split.y)
    for epoch in range(EPOCHS):
        order = rng.permutation(n_samples)
        total_loss = 0.0
        for start in range(0, n_samples, BATCH_SIZE):
            batch_idx = torch.from_numpy(order[start : start + BATCH_SIZE])
            optimizer.zero_grad()
            pred = model(train_x[batch_idx])
            loss = loss_fn(pred, train_y[batch_idx])
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(batch_idx)

        epoch_loss = total_loss / n_samples
        losses.append(epoch_loss)
        print(f"epoch {epoch + 1:2d}/{EPOCHS} loss={epoch_loss:.8f}")

    model.eval()
    return model, losses


@torch.no_grad()
def predict_scores(model, features):
    model.eval()
    x_tensor = torch.from_numpy(features)
    logits = model(x_tensor)
    probs = torch.sigmoid(logits).squeeze(1)
    return probs.numpy()


def save_training_outputs(out_dir, model, splits, scaler, source_csv, losses):
    out_dir.mkdir(parents=True, exist_ok=True)
    train_s, val_s, test_s = splits

    val_scores = predict_scores(model, val_s.x)
    test_scores = predict_scores(model, test_s.x)

    torch.save(model.state_dict(), out_dir / "model.pt")

    # зберігаємо масиви для подальшого аналізу без повторного навчання
    np.savez(
        out_dir / "split_and_predictions.npz",
        train_indices=train_s.row_ids,
        validation_indices=val_s.row_ids,
        test_indices=test_s.row_ids,
        train_labels=train_s.y,
        validation_labels=val_s.y,
        test_labels=test_s.y,
        validation_scores=val_scores,
        test_scores=test_scores,
        train_amount=train_s.amount,
        validation_amount=val_s.amount,
        test_amount=test_s.amount,
        scaler_mean=scaler.mean_,
        scaler_scale=scaler.scale_,
    )

    meta = {
        "source_csv": source_csv.name,
        "features": FEATURES,
        "seed": SEED,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "architecture": [len(FEATURES), HIDDEN_UNITS, 1],
        "dtype": "float32",
        "device": "cpu",
        "loss_history": losses,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "torch": str(torch.__version__),
    }
    with open(out_dir / "run.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
        f.write("\n")


def train_from_csv(csv_path, out_dir):
    train_s, val_s, test_s, scaler = make_splits(csv_path)
    model, losses = train_model(train_s)
    save_training_outputs(out_dir, model, (train_s, val_s, test_s), scaler, csv_path, losses)
    return {
        "splits": split_summary({
            "train_labels": train_s.y,
            "validation_labels": val_s.y,
            "test_labels": test_s.y,
        }),
        "loss_history": losses,
    }


def load_saved_predictions(out_dir):
    p1 = out_dir / "split_and_predictions.npz"
    p2 = out_dir / "predictions.npz"
    path = p1 if p1.exists() else p2
    if not path.exists():
        raise FileNotFoundError(f"Не знайдено файлу прогнозів у {out_dir}")

    with np.load(path, allow_pickle=False) as data:
        res = {k: data[k] for k in data.files}

    if "validation_labels" in res:
        return res

    # сумісність із назвами полів зі стартового коду
    return {
        "train_indices": res["train_row_ids"],
        "validation_indices": res["validation_row_ids"],
        "test_indices": res["test_row_ids"],
        "train_labels": res["train_y"],
        "validation_labels": res["validation_y"],
        "test_labels": res["test_y"],
        "validation_scores": res["validation_scores"],
        "test_scores": res["test_scores"],
        "validation_amount": res["validation_amount"],
        "test_amount": res["test_amount"],
    }


def split_summary(data):
    rows = []
    for name in ("train", "validation", "test"):
        lbls = data[f"{name}_labels"]
        pos = int(np.sum(lbls == 1))
        rows.append({
            "split": name,
            "n": len(lbls),
            "positives": pos,
            "positive_fraction": pos / len(lbls),
        })
    return pd.DataFrame(rows)


def counts(labels, scores, threshold):
    y = np.asarray(labels, dtype=np.int64)
    pred = np.asarray(scores) >= threshold
    tp = int(np.sum((pred == 1) & (y == 1)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    c = 10 * fn + fp
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn, "C": c}


def candidate_table(labels, scores):
    y = np.asarray(labels, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)

    # сортуємо за спаданням оцінок
    order = np.argsort(-s, kind="stable")
    sorted_s = s[order]
    sorted_y = y[order]

    # знаходимо останні індекси для кожної групи однакових оцінок
    is_last = np.empty(len(s), dtype=bool)
    is_last[:-1] = (sorted_s[:-1] != sorted_s[1:])
    is_last[-1] = True
    last_idx = np.where(is_last)[0]

    # рахуємо кумулятивну суму позитивних міток
    cum_tp = np.cumsum(sorted_y)
    tp_vals = cum_tp[last_idx]
    fp_vals = (last_idx + 1) - tp_vals

    total_p = int(y.sum())
    total_n = len(y) - total_p

    # додаємо поріг нескінченність коли всі прогнози негативні
    thresholds = np.concatenate([[np.inf], sorted_s[last_idx]])
    tp = np.concatenate([[0], tp_vals])
    fp = np.concatenate([[0], fp_vals])
    fn = total_p - tp
    tn = total_n - fp
    c = 10 * fn + fp

    return pd.DataFrame({
        "threshold": thresholds,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "C": c,
    })


def best_threshold(table):
    # шукаємо найменшу вартість
    min_cost = table["C"].min()
    # якщо є кілька однакових мінімумів, беремо найбільший поріг
    best_rows = table[table["C"] == min_cost]
    return float(best_rows["threshold"].max())


def verify_cases():
    # тестові набори з умови завдання
    cases = {
        "A": ([0.9, 0.7, 0.7, 0.4, 0.2, 0.1], [0, 1, 0, 1, 0, 0]),
        "B": ([0.9, 0.6, 0.3], [0, 0, 0]),
        "C": ([0.8] + [0.5] * 11 + [0.2], [1, 1] + [0] * 10 + [0]),
    }
    res = {}
    for name, (sc, lb) in cases.items():
        # швидкий розрахунок через кумулятивні суми
        fast = candidate_table(lb, sc)
        # повільний прямий перебір для перевірки
        cands = [np.inf] + sorted(set(sc), reverse=True)
        slow = pd.DataFrame([{"threshold": t, **counts(lb, sc, t)} for t in cands])
        pd.testing.assert_frame_equal(fast, slow, check_dtype=False)

        slow_min = slow["C"].min()
        slow_best = float(slow[slow["C"] == slow_min]["threshold"].max())
        fast_best = best_threshold(fast)
        assert fast_best == slow_best, name
        res[name] = (fast, fast_best)
    return res


def _format_thresh(t):
    return "+inf" if np.isposinf(t) else float(t)


def evaluate_threshold(labels, scores, threshold):
    res = {"threshold": _format_thresh(threshold)}
    res.update(counts(labels, scores, threshold))
    # обчислюємо точність та повноту
    if res["TP"] + res["FP"] > 0:
        res["precision"] = res["TP"] / (res["TP"] + res["FP"])
    else:
        res["precision"] = None

    if res["TP"] + res["FN"] > 0:
        res["recall"] = res["TP"] / (res["TP"] + res["FN"])
    else:
        res["recall"] = None
    return res


def error_table(data, threshold):
    labels = data["test_labels"]
    scores = data["test_scores"]
    indices = data["test_indices"]
    amounts = data["test_amount"]

    pred = scores >= threshold
    rows = []

    # перші п'ять хибнопозитивних прикладів
    fp_mask = (pred == 1) & (labels == 0)
    fp_idx = np.where(fp_mask)[0]
    # сортуємо за номером рядка у вихідному файлі
    fp_sorted = fp_idx[np.argsort(indices[fp_idx])]
    for i in fp_sorted[:5]:
        rows.append({
            "csv_row": int(indices[i]),
            "error": "FP",
            "label": int(labels[i]),
            "score": float(scores[i]),
            "Amount": float(amounts[i]),
            "score_minus_threshold": float(scores[i] - threshold),
        })

    # перші п'ять хибнонегативних прикладів
    fn_mask = (pred == 0) & (labels == 1)
    fn_idx = np.where(fn_mask)[0]
    fn_sorted = fn_idx[np.argsort(indices[fn_idx])]
    for i in fn_sorted[:5]:
        rows.append({
            "csv_row": int(indices[i]),
            "error": "FN",
            "label": int(labels[i]),
            "score": float(scores[i]),
            "Amount": float(amounts[i]),
            "score_minus_threshold": float(scores[i] - threshold),
        })

    return pd.DataFrame(rows)


def save_validation_curves(out_dir, labels, scores):
    prec, rec, _ = precision_recall_curve(labels, scores)
    fpr, tpr, _ = roc_curve(labels, scores)

    # малюємо криву точність повнота
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(rec, prec, linewidth=1.5)
    ax.set(title="Validation precision-recall curve", xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1))
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "validation_pr.png", dpi=160)
    plt.close(fig)

    # малюємо roc криву
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, linewidth=1.5)
    ax.set(title="Validation ROC curve", xlabel="False positive rate", ylabel="True positive rate", xlim=(0, 1), ylim=(0, 1))
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "validation_roc.png", dpi=160)
    plt.close(fig)

    return float(auc(fpr, tpr))


def analyze_saved(out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    data = load_saved_predictions(out_dir)

    # спочатку запускаємо перевірки на малих наборах
    verif = verify_cases()
    verif_res = {}
    for name, (tbl, t) in verif.items():
        tbl.to_csv(out_dir / f"verification_{name}.csv", index=False, float_format="%.17g")
        verif_res[name] = _format_thresh(t)

    val_y = data["validation_labels"]
    val_s = data["validation_scores"]
    test_y = data["test_labels"]
    test_s = data["test_scores"]

    # шукаємо оптимальний поріг на валідації
    candidates = candidate_table(val_y, val_s)
    candidates.to_csv(out_dir / "validation_candidates.csv", index=False, float_format="%.17g")
    best_t = best_threshold(candidates)
    with open(out_dir / "chosen_threshold.txt", "w", encoding="utf-8") as f:
        f.write(f"{best_t:.17g}\n")

    roc_auc = save_validation_curves(out_dir, val_y, val_s)

    val_summary = {
        "standard_0.5": evaluate_threshold(val_y, val_s, 0.5),
        "minimum_cost": evaluate_threshold(val_y, val_s, best_t),
        "all_negative": evaluate_threshold(val_y, val_s, float("inf")),
    }

    test_summary = evaluate_threshold(test_y, test_s, best_t)
    errors = error_table(data, best_t)
    errors.to_csv(out_dir / "test_errors.csv", index=False, float_format="%.17g")

    sp_df = split_summary(data)
    sp_dict = {
        r["split"]: {
            "total": int(r["n"]),
            "fraud": int(r["positives"]),
            "fraud_fraction": float(r["positive_fraction"]),
        }
        for r in sp_df.to_dict("records")
    }

    summary = {
        "splits": sp_dict,
        "roc_auc": roc_auc,
        "selected_threshold": best_t,
        "validation": val_summary,
        "verification": verif_res,
        "test": test_summary,
        "errors": errors.to_dict("records"),
    }

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")

    return summary


def make_parser():
    parser = argparse.ArgumentParser(description="Лабораторна робота 2")
    subs = parser.add_subparsers(dest="command", required=True)

    p_train = subs.add_parser("train", help="навчання моделі")
    p_train.add_argument("--csv", type=Path, required=True, help="шлях до creditcard.csv")
    p_train.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "artifacts")

    p_an = subs.add_parser("analyze", help="аналіз прогнозів")
    p_an.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "artifacts")

    p_run = subs.add_parser("run", help="навчання та аналіз разом")
    p_run.add_argument("--csv", type=Path, required=True, help="шлях до creditcard.csv")
    p_run.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "artifacts")

    return parser


def main(argv=None):
    parser = make_parser()
    args = parser.parse_args(argv)
    if args.command == "train":
        res = train_from_csv(args.csv, args.out)
        print(res["splits"].to_string(index=False))
        print(f"Збережено у {args.out.resolve()}")
    elif args.command == "analyze":
        s = analyze_saved(args.out)
        print(json.dumps(s, ensure_ascii=False, indent=2, allow_nan=False))
    else:
        train_from_csv(args.csv, args.out)
        s = analyze_saved(args.out)
        print(json.dumps(s, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
