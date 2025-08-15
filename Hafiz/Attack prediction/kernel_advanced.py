import numpy as np
import cv2
import tensorflow as tf
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.losses import CategoricalCrossentropy
import matplotlib.pyplot as plt

IMAGE_PATH = "why.jpg"
TARGET_SIZE = (224, 224)
EPSILONS = [0.5, 1.0, 2.0]
DEFENSES = {
    "avg3x3": lambda im: cv2.blur(im, (3, 3)),
    "median5x5": lambda im: cv2.medianBlur(im, 5),
    "gauss5x5": lambda im: cv2.GaussianBlur(im, (5, 5), sigmaX=1.0)
}

def load_and_preprocess(path):
    img = load_img(path, target_size=TARGET_SIZE)
    arr = img_to_array(img)
    x = np.expand_dims(arr.copy(), axis=0)
    return arr.astype(np.uint8), tf.convert_to_tensor(preprocess_input(x))

@tf.function
def fgsm_attack(model, x, epsilon):
    with tf.GradientTape() as tape:
        tape.watch(x)
        logits = model(x)
        label = tf.one_hot(tf.argmax(logits[0]), 1000)[None, :]
        loss = CategoricalCrossentropy()(label, logits)
    grad = tape.gradient(loss, x)
    return tf.clip_by_value(x + epsilon * tf.sign(grad), -1.0, 1.0)

@tf.function
def tf_blur_defense(adv_x, k=3):
    kernel = tf.ones((k, k, 3, 1), dtype=tf.float32) / (k * k)
    return tf.nn.depthwise_conv2d(adv_x, kernel, strides=[1,1,1,1], padding="SAME")


def predict_and_decode(model, x):
    preds = model.predict(x)
    return decode_predictions(preds, top=1)[0][0]


def viterbi(obs_seq, confidences, states, start_p, trans_p):
    T, N = len(obs_seq), len(states)
    emit_p = np.zeros((T, N))
    for t in range(T):
        for i, st in enumerate(states):
            emit_p[t, i] = confidences[t] if st == obs_seq[t] else (1 - confidences[t]) / (N - 1)
    V = np.zeros((T, N))
    backpointer = np.zeros((T, N), dtype=int)
    V[0] = start_p * emit_p[0]
    for t in range(1, T):
        for j in range(N):
            seq_probs = V[t-1] * trans_p[:, j] * emit_p[t, j]
            backpointer[t, j] = np.argmax(seq_probs)
            V[t, j] = np.max(seq_probs)
    best_path = np.zeros(T, dtype=int)
    best_path[-1] = np.argmax(V[-1])
    for t in range(T-2, -1, -1):
        best_path[t] = backpointer[t+1, best_path[t+1]]
    return [states[i] for i in best_path], states[best_path[-1]]

def main():
    model = VGG16(weights="imagenet")
    model.trainable = False

    rgb_raw, x_clean = load_and_preprocess(IMAGE_PATH)
    clean_id, clean_label, clean_conf = predict_and_decode(model, x_clean)
    print(f"Clean prediction: {clean_label} ({clean_conf:.2f})")

    all_results = {}
    for eps in EPSILONS:
        adv = fgsm_attack(model, x_clean, eps)
        _, adv_label, adv_conf = predict_and_decode(model, adv)
        print(f"ε={eps}  Adversarial: {adv_label} ({adv_conf:.2f})")

        adv_uint8 = ((adv[0].numpy() + 1) * 127.5).astype(np.uint8)
        obs_labels, obs_confs = [], []

        for name, fn in DEFENSES.items():
            img_def = fn(adv_uint8)
            _, lab, conf = predict_and_decode(model, preprocess_input(img_def.astype(np.float32)[None]))
            print(f"  [{name}] {lab} ({conf:.2f})")
            obs_labels.append(lab)
            obs_confs.append(conf)

        tf_def = tf_blur_defense(adv, k=3)
        tf_img = ((tf_def[0].numpy() + 1) * 127.5).astype(np.uint8)
        _, tf_lab, tf_conf = predict_and_decode(model, preprocess_input(tf_img.astype(np.float32)[None]))
        print(f"  [tf_avg3x3] {tf_lab} ({tf_conf:.2f})")
        obs_labels.append(tf_lab)
        obs_confs.append(tf_conf)

        states = list(set(obs_labels))
        N = len(states)
        start_p = np.ones(N) / N
        trans_p = np.full((N, N), 0.1 / (N - 1))
        np.fill_diagonal(trans_p, 0.9)
        _, hmm_label = viterbi(obs_labels, obs_confs, states, start_p, trans_p)
        print(f"  [HMM smoothed] {hmm_label}")

        all_results[eps] = {
            "adv": (adv_label, adv_conf),
            **{n: (l, c) for n, l, c in zip(list(DEFENSES.keys())+['tf_avg3x3'], obs_labels, obs_confs)},
            "hmm": hmm_label
        }

    eps = EPSILONS[-1]
    adv = fgsm_attack(model, x_clean, eps)
    adv_uint8 = ((adv[0].numpy()+1)*127.5).astype(np.uint8)
    cols = len(DEFENSES) + 4  # clean, adv, defenses, tf, hmm
    fig, axes = plt.subplots(1, cols, figsize=(4*cols, 4))

    axes[0].imshow(rgb_raw)
    axes[0].set_title(f"Clean\n{clean_label} ({clean_conf:.2f})")
    axes[1].imshow(adv_uint8)
    a_lab, a_conf = all_results[eps]['adv']
    axes[1].set_title(f"Adv ε={eps}\n{a_lab} ({a_conf:.2f})")

    idx = 2
    for name in DEFENSES:
        img_def = DEFENSES[name](adv_uint8)
        lab, conf = all_results[eps][name]
        axes[idx].imshow(img_def)
        axes[idx].set_title(f"{name}\n{lab} ({conf:.2f})")
        idx += 1

    axes[idx].imshow(tf_img)
    lab, conf = all_results[eps]['tf_avg3x3']
    axes[idx].set_title(f"tf_avg3x3\n{lab} ({conf:.2f})")
    idx += 1

    axes[idx].axis('off')
    axes[idx].set_title(f"HMM\n{all_results[eps]['hmm']}")

    for ax in axes:
        ax.axis('off')
    plt.tight_layout()
    plt.savefig("adv_defense_hmm.png")
    plt.show()

if __name__ == "__main__":
    main()
