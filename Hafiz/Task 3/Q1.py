import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('why.jpg', cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("Put a valid image at 'why.jpg'")

K_sharp = np.array([[ 0, -1,  0],
                    [-1,  5, -1],
                    [ 0, -1,  0]], dtype=np.float32)

K_edge  = np.array([[-1, -2, -1],
                    [ 0,  0,  0],
                    [ 1,  2,  1]], dtype=np.float32)

J_sharp = cv2.filter2D(img, ddepth=-1, kernel=K_sharp, borderType=cv2.BORDER_CONSTANT)
J_edge  = cv2.filter2D(img, ddepth=-1, kernel=K_edge,  borderType=cv2.BORDER_CONSTANT)

fig, axes = plt.subplots(2, 3, figsize=(12, 8))

axes[0,0].imshow(img,      cmap='gray'); axes[0,0].set_title('Original');     axes[0,0].axis('off')
axes[0,1].imshow(J_sharp,  cmap='gray'); axes[0,1].set_title('Sharpened');    axes[0,1].axis('off')
axes[0,2].imshow(J_edge,   cmap='gray'); axes[0,2].set_title('Edges');        axes[0,2].axis('off')

axes[1,0].hist(img.ravel(),     bins=256); axes[1,0].set_title('Hist: Original')
axes[1,1].hist(J_sharp.ravel(), bins=256); axes[1,1].set_title('Hist: Sharpened')
axes[1,2].hist(J_edge.ravel(),  bins=256); axes[1,2].set_title('Hist: Edges')

plt.tight_layout()
plt.show()
