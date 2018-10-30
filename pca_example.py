from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

plt.gray()
plt.figure(figsize=(8, 4))

digits = load_digits()
images = digits.images

X = np.reshape(images, (len(images), -1))

# Original Image
plt.subplot(1, 3, 1);
plt.imshow(X[0].reshape(8, 8))
plt.title("Original Image")

# from 8*8 to 6*6
pca1 = PCA(n_components=6*6)
pca1.fit(X)
X_new1 = pca1.transform(X)
X_approx1 = pca1.inverse_transform(X_new1)

plt.subplot(1, 3, 2);
plt.imshow(X_approx1[0].reshape(8, 8))
plt.title("36 components")

# from 8*8 to 4*4
pca2 = PCA(n_components=4*4)
pca2.fit(X)
X_new2 = pca2.transform(X)
X_approx2 = pca2.inverse_transform(X_new2)

plt.subplot(1, 3, 3);
plt.imshow(X_approx2[0].reshape(8, 8))
plt.title("28 components")

plt.matshow(digits.images[0])
plt.show()