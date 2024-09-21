import sklearn
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import numpy as np

# Carica il dataset MNIST
mnist = fetch_openml('mnist_784', as_frame=False, parser='liac-arff')
Xraw, Yraw = mnist['data'], mnist['target']

# Non è necessario convertire Xraw, è già un array NumPy
# Xraw = Xraw.to_numpy()  # Questa riga può essere rimossa
Yraw = Yraw.astype(int)

# Visualizza le prime 5 righe di Xraw e le prime 5 etichette in Yraw
print("First 5 rows of Xraw:")
print(Xraw[:5])
print("First 5 labels in Yraw:")
print(Yraw[:5])

# Visualizza la prima immagine (opzionale)
first_image = Xraw[0].reshape(28, 28)  # MNIST images are 28x28 pixels
plt.imshow(first_image, cmap='gray')
plt.title(f"Label: {Yraw[0]}")
plt.show()

