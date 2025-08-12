import argparse
import logging
from pathlib import Path
from typing import Callable, Dict, List, Tuple, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import yaml
from keras.applications import VGG16
from keras.applications.vgg16 import decode_predictions, preprocess_input
from keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.losses import CategoricalCrossentropy
from tqdm import tqdm

AttackFn = Callable[[tf.keras.Model, tf.Tensor, floxat], tf.Tensor]
DefenseFn = Callable[[np.ndarray], np.ndarray]


def load_config(path: Path) -> Dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_model(name: str = "imagenet", log_dir: Optional[Path] = None) -> tf.keras.Model:
    model = VGG16(weights=name)
    model.trainable = False
    if log_dir:
        tf.keras.utils.plot_model(model, to_file=str(log_dir / "model.png"), show_shapes=True)
    return model


def load_image(path: Path, size: Tuple[int, int]) -> Tuple[np.ndarray, tf.Tensor]:
    img = load_img(str(path), target_size=size)
    arr = img_to_array(img).astype(np.uint8)
    x = np.expand_dims(arr.astype(np.float32), axis=0)
    return arr, tf.convert_to_tensor(preprocess_input(x))


@tf.function
def fgsm(model: tf.keras.Model, x: tf.Tensor, eps: float) -> tf.Tensor:
    with tf.GradientTape() as tape:
        tape.watch(x)
        logits = model(x)
        label = tf.one_hot(tf.argmax(logits[0]), logits.shape[-1])[None, :]
        loss = CategoricalCrossentropy()(label, logits)
    grad = tape.gradient(loss, x)
    return tf.clip_by_value(x + eps * tf.sign(grad), -1.0, 1.0)


@tf.function
def pgd(model: tf.keras.Model, x: tf.Tensor, eps: float, alpha: float, iters: int) -> tf.Tensor:
    delta = tf.zeros_like(x)
    for _ in range(iters):
        with tf.GradientTape() as tape:
            tape.watch(delta)
            logits = model(x + delta)
            label = tf.one_hot(tf.argmax(logits[0]), logits.shape[-1])[None, :]
            loss = CategoricalCrossentropy()(label, logits)
        grad = tape.gradient(loss, delta)
        delta = tf.clip_by_value(delta + alpha * tf.sign(grad), -eps, eps)
    return tf.clip_by_value(x + delta, -1.0, 1.0)


def standard_defenses() -> Dict[str, DefenseFn]:
    return {
        "avg3x3": lambda im: cv2.blur(im, (3, 3)),
        "median5x5": lambda im: cv2.medianBlur(im, 5),
        "gauss5x5": lambda im: cv2.GaussianBlur(im, (5, 5), 1.0),
    }


def predict(model: tf.keras.Model, x: tf.Tensor) -> Tuple[str, float]:
    preds = model.predict(x, verbose=0)
    _, label, conf = decode_predictions(preds, top=1)[0][0]
    return label, float(conf)


def build_viterbi(states: List[str], prior: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    N = len(states)
    start_p = np.ones(N) / N
    trans_p = np.full((N, N), prior / (N - 1))
    np.fill_diagonal(trans_p, 1 - prior + prior / (N - 1))
    return start_p, trans_p


def viterbi(obs: List[str], confs: List[float], states: List[str],
            start_p: np.ndarray, trans_p: np.ndarray) -> str:
    T, N = len(obs), len(states)
    emit = np.zeros((T, N))
    for t in range(T):
        for i, s in enumerate(states):
            emit[t, i] = confs[t] if s == obs[t] else (1 - confs[t]) / (N - 1)

    V = np.zeros((T, N))
    bp = np.zeros((T, N), dtype=int)
    V[0] = start_p * emit[0]
    for t in range(1, T):
        for j in range(N):
            probs = V[t - 1] * trans_p[:, j] * emit[t, j]
            bp[t, j] = np.argmax(probs)
            V[t, j] = np.max(probs)
    last = np.argmax(V[-1])
    for t in range(T - 1, 0, -1):
        last = bp[t, last]
    return states[last]


def run_attack_defense(
    model: tf.keras.Model,
    img_tensor: tf.Tensor,
    raw_img: np.ndarray,
    eps: float,
    atk_cfg: Dict,
    defenses: Dict[str, DefenseFn],
    start_p: np.ndarray,
    trans_p: np.ndarray
) -> Dict:
    atk_fn: AttackFn = fgsm if atk_cfg["type"] == "fgsm" else (
        lambda m, x, e: pgd(m, x, e, atk_cfg["alpha"], atk_cfg["iters"]))
    adv = atk_fn(model, img_tensor, eps)
    adv_arr = adv[0].numpy()
    label_adv, conf_adv = predict(model, adv)
    obs_labels, obs_confs = [], []
    proc_imgs = {}

    for name, fn in defenses.items():
        proc = fn(((adv_arr + 1) * 127.5).astype(np.uint8))
        proc_imgs[name] = proc
        inp = tf.expand_dims(preprocess_input(proc.astype(np.float32)), axis=0)
        l, c = predict(model, inp)
        obs_labels.append(l)
        obs_confs.append(c)

    hmm_label = viterbi(obs_labels, obs_confs, list(defenses.keys()), start_p, trans_p)
    return {
        "eps": eps,
        "adv": (label_adv, conf_adv),
        "defs": list(zip(defenses.keys(), obs_labels, obs_confs)),
        "smooth": hmm_label,
        "adv_img": adv_arr,
        "proc_imgs": proc_imgs
    }


def plot_and_save(results: List[Dict], raw: np.ndarray, out_path: Path) -> None:
    rows = len(results)
    cols = max(3, len(results[0]["defs"]) + 2)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    for i, res in enumerate(results):
        axes[i, 0].imshow(raw)
        axes[i, 0].set_title("Clean")
        axes[i, 0].axis("off")

        adv_uint = ((res["adv_img"] + 1) * 127.5).astype(np.uint8)
        axes[i, 1].imshow(adv_uint)
        a_lab, a_conf = res["adv"]
        axes[i, 1].set_title(f"Adv ε={res['eps']}\n{a_lab} ({a_conf:.2f})")
        axes[i, 1].axis("off")

        for j, (name, lbl, conf) in enumerate(res["defs"], start=2):
            img = res["proc_imgs"][name]
            axes[i, j].imshow(img)
            axes[i, j].set_title(f"{name}\n{lbl} ({conf:.2f})")
            axes[i, j].axis("off")

        axes[i, -1].axis("off")
        axes[i, -1].set_title(f"HMM: {res['smooth']}")

    plt.tight_layout()
    plt.savefig(str(out_path))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)

    log_dir = Path(cfg.get("log_dir", "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=log_dir / "run.log",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s"
    )

    tb = TensorBoard(log_dir=str(log_dir / "tensorboard"))
    model = build_model(cfg.get("model", "imagenet"), log_dir)
    raw, tensor = load_image(Path(cfg["image"]), tuple(cfg.get("size", [224, 224])))
    clean_lbl, clean_conf = predict(model, tensor)
    logging.info(f"Clean Prediction: {clean_lbl} ({clean_conf:.2f})")

    defenses = standard_defenses()
    start_p, trans_p = build_viterbi(list(defenses.keys()), cfg.get("hmm_prior", 0.1))

    results = []
    for eps in tqdm(cfg["epsilons"], desc="Attacks"):
        res = run_attack_defense(
            model, tensor, raw, eps, cfg["attack"], defenses, start_p, trans_p
        )
        results.append(res)
        logging.info(f"ε={eps} Result: {res}")

    plot_and_save(results, raw, Path(cfg.get("output", "results.png")))


if __name__ == "__main__":
    main()