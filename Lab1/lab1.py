# Лабораторна робота з глибокого навчання (DL)
# Студент гр. КМ-32 Латко Назар Андрійович
# Тема: Реалізація та перевірка зворотного поширення похибки

import argparse
from dataclasses import dataclass
import numpy as np
import torch
from sklearn.datasets import load_iris


# параметри для перевірки числового градієнта.
PARAMETER_SPECS = (("W1", (0, 0)), ("b1", (0,)), ("W2", (0, 0)), ("b2", (0,)))


@dataclass
class Parameters:
    W1: np.ndarray
    b1: np.ndarray
    W2: np.ndarray
    b2: np.ndarray


def prepare_data():
    # завантажуємо датасет ірисів.
    iris = load_iris()
    features = np.asarray(iris.data, dtype=np.float64)
    labels = np.asarray(iris.target, dtype=np.int64)

    # фіксований генератор для відтворюваності розбиття.
    rng = np.random.default_rng(0)
    train_idx = []
    test_idx = []

    # беремо по 35 обєктів кожного класу у навчання, а решту 15 у тест.
    for c in range(3):
        idx = np.flatnonzero(labels == c)
        rng.shuffle(idx)
        train_idx.extend(idx[:35])
        test_idx.extend(idx[35:])

    train_idx = np.array(train_idx)
    test_idx = np.array(test_idx)

    # рахуємо середнє і стандартне відхилення тільки на навчальній вибірці.
    mean = features[train_idx].mean(axis=0)
    std = features[train_idx].std(axis=0, ddof=0)

    # стандартизуємо обидві вибірки.
    X_train = (features[train_idx] - mean) / std
    X_test = (features[test_idx] - mean) / std

    return X_train, labels[train_idx], X_test, labels[test_idx]


def initialize() -> Parameters:
    # ініціалізація ваг за правилами хе та ксавє.
    rng = np.random.default_rng(0)
    W1 = rng.normal(0.0, np.sqrt(2.0 / 4.0), size=(4, 8))
    W2 = rng.normal(0.0, np.sqrt(2.0 / (8.0 + 3.0)), size=(8, 3))
    b1 = np.zeros(8, dtype=np.float64)
    b2 = np.zeros(3, dtype=np.float64)
    return Parameters(W1, b1, W2, b2)


def forward(
    X: np.ndarray,
    y: np.ndarray,
    parameters: Parameters,
) -> tuple[float, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    # перший шар і активація релу.
    z1 = X @ parameters.W1 + parameters.b1
    a1 = np.maximum(z1, 0.0)

    # другий шар, логіти.
    logits = a1 @ parameters.W2 + parameters.b2

    # віднімаємо максимум для числової стабільності експоненти.
    shifted_logits = logits - logits.max(axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(
        np.exp(shifted_logits).sum(axis=1, keepdims=True)
    )

    # середня крос ентропія за навчальною вибіркою.
    loss = -log_probs[np.arange(len(y)), y].mean()

    # повертаємо втрату та проміжні значення для зворотного ходу.
    return float(loss), (z1, a1, np.exp(log_probs))


def backward(
    X: np.ndarray,
    y: np.ndarray,
    parameters: Parameters,
    cache: tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    bug: bool = False,
) -> Parameters:
    z1, a1, probabilities = cache
    n = len(y)

    # градієнт функції втрат за логітами.
    dlogits = probabilities.copy()
    dlogits[np.arange(n), y] -= 1.0

    # у досліді з помилкою навмисно пропускаємо ділення на кількість прикладів.
    if not bug:
        dlogits /= n

    # градієнти другого лінійного шару.
    dW2 = a1.T @ dlogits
    db2 = dlogits.sum(axis=0)

    # градієнти першого шару з урахуванням похідної релу.
    dz1 = (dlogits @ parameters.W2.T) * (z1 > 0.0)
    dW1 = X.T @ dz1
    db1 = dz1.sum(axis=0)

    return Parameters(dW1, db1, dW2, db2)


def torch_reference(
    X: np.ndarray,
    y: np.ndarray,
    parameters: Parameters,
) -> tuple[float, Parameters]:
    # еталонна мережа на пайторч для перевірки.
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 8, dtype=torch.float64),
        torch.nn.ReLU(),
        torch.nn.Linear(8, 3, dtype=torch.float64),
    )

    # копіюємо параметри у модель, транспонуючи ваги через особливість лінійного шару пайторч.
    with torch.no_grad():
        model[0].weight.copy_(torch.from_numpy(parameters.W1.T.copy()))
        model[0].bias.copy_(torch.from_numpy(parameters.b1))
        model[2].weight.copy_(torch.from_numpy(parameters.W2.T.copy()))
        model[2].bias.copy_(torch.from_numpy(parameters.b2))

    loss = torch.nn.functional.cross_entropy(
        model(torch.from_numpy(X)), torch.from_numpy(y)
    )
    loss.backward()

    gradients = Parameters(
        model[0].weight.grad.detach().numpy().T.copy(),
        model[0].bias.grad.detach().numpy().copy(),
        model[2].weight.grad.detach().numpy().T.copy(),
        model[2].bias.grad.detach().numpy().copy(),
    )
    return float(loss.item()), gradients


def finite_difference(
    X: np.ndarray,
    y: np.ndarray,
    parameters: Parameters,
    name: str,
    index: tuple[int, ...],
    epsilon: float = 1e-6,
) -> float:
    # центральна числова похідна для одного конкретного параметра.
    array = getattr(parameters, name)
    orig = array[index]

    array[index] = orig + epsilon
    loss_plus, _ = forward(X, y, parameters)

    array[index] = orig - epsilon
    loss_minus, _ = forward(X, y, parameters)

    # повертаємо старе значення назад.
    array[index] = orig

    return (loss_plus - loss_minus) / (2.0 * epsilon)


def forward_logits(X: np.ndarray, parameters: Parameters) -> np.ndarray:
    # прямий прохід тільки для отримання логітів і точності.
    h = np.maximum(X @ parameters.W1 + parameters.b1, 0.0)
    return h @ parameters.W2 + parameters.b2


def compare_gradients(
    gradients: Parameters,
    reference_gradients: Parameters,
) -> dict[str, tuple[float, bool]]:
    # порівняння знайдених вручну градієнтів з пайторч.
    result = {}
    for name in ("W1", "b1", "W2", "b2"):
        diff = np.abs(getattr(gradients, name) - getattr(reference_gradients, name))
        passed = bool(np.isfinite(diff).all() and (diff <= 1e-12).all())
        result[name] = (float(diff.max()), passed)
    return result


def finite_difference_checks(
    X: np.ndarray,
    y: np.ndarray,
    parameters: Parameters,
    gradients: Parameters,
) -> list[tuple[str, float, float, float, bool]]:
    # розрахунок числової похідної для чотирьох заданих елементів.
    rows = []
    for name, index in PARAMETER_SPECS:
        analytic = float(getattr(gradients, name)[index])
        numerical = finite_difference(X, y, parameters, name, index)
        diff = abs(analytic - numerical)
        passed = bool(np.isfinite(diff) and diff <= 1e-7)
        rows.append((f"{name}{index}", analytic, numerical, diff, passed))
    return rows


def run(bug: bool = False) -> None:
    # запуск розрахунків і вивід результатів.
    X, y, X_test, y_test = prepare_data()
    parameters = initialize()

    loss, cache = forward(X, y, parameters)
    gradients = backward(X, y, parameters, cache, bug=bug)
    ref_loss, ref_grads = torch_reference(X, y, parameters)

    print(f"mode={'bug' if bug else 'correct'} samples_train={len(y)} samples_test={len(y_test)}")
    print(f"loss_numpy={loss:.16g} loss_torch={ref_loss:.16g}")
    print("PyTorch comparison (threshold: 1e-12)")
    print("quantity | max_absolute_difference | passed")

    loss_diff = abs(loss - ref_loss)
    loss_passed = bool(np.isfinite(loss_diff) and loss_diff <= 1e-12)
    print(f"loss | {loss_diff:.12g} | {loss_passed}")

    for name, (diff, passed) in compare_gradients(gradients, ref_grads).items():
        print(f"{name} | {diff:.12g} | {passed}")

    print("Finite differences (threshold: 1e-7)")
    print("parameter | backward | numerical | absolute_difference | passed")
    for name, analytic, numerical, diff, passed in finite_difference_checks(
        X, y, parameters, gradients
    ):
        print(
            f"{name} | {analytic:.12g} | {numerical:.12g} | "
            f"{diff:.12g} | {passed}"
        )

    test_preds = np.argmax(forward_logits(X_test, parameters), axis=1)
    test_acc = np.mean(test_preds == y_test)
    print(f"initial_test_accuracy={test_acc:.12g}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bug",
        action="store_true",
        help="запуск з навмисною помилкою без ділення на n",
    )
    args = parser.parse_args()
    run(args.bug)
