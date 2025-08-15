import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.preprocessing.image import load_img, img_to_array
import cv2
model = VGG16(weights="imagenet")
model.trainable = False
img = load_img("why.jpg", target_size=(224, 224))
img_rgb = np.array(img)
x_raw = img_to_array(img)
x_exp = np.expand_dims(x_raw, axis=0)
x = tf.Variable(preprocess_input(x_exp), dtype=tf.float32)

pred = model.predict(x)

loss_object = tf.keras.losses.CategoricalCrossentropy()
with tf.GradientTape() as tape:
    tape.watch(x)
    prediction = model(x)
    label = tf.one_hot(tf.argmax(prediction[0]), 1000)
    label = tf.reshape(label, (1, 1000))
    loss = loss_object(label, prediction)

gradient = tape.gradient(loss, x)
epsilon = 2.0
signed_grad = tf.sign(gradient)
adv_x = x + epsilon * signed_grad
adv_x = tf.clip_by_value(adv_x, -1.0, 1.0)

adv_pred = model.predict(adv_x)
print("Adversarial prediction:", decode_predictions(adv_pred, top=1)[0])

adv_img_np = adv_x[0].numpy()
adv_img_uint8 = ((adv_img_np + 1) * 127.5).astype(np.uint8)
smoothed = cv2.blur(adv_img_uint8, ksize=(3, 3))

smoothed_input = preprocess_input(np.expand_dims(smoothed.astype(np.float32), axis=0))
recovered_pred = model.predict(smoothed_input)
print("🛡️ Recovered prediction after smoothing:", decode_predictions(recovered_pred, top=1)[0])

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(img_rgb.astype(np.uint8))
axes[0].set_title("Original")
axes[1].imshow(adv_img_uint8)
axes[1].set_title("Adversarial")
axes[2].imshow(smoothed)
axes[2].set_title("After Kernel Defense")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.savefig("defense_result.png")
plt.show()
