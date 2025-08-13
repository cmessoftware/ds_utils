import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np

class Viz:
   def plot_matriz_confusion(y_true, y_pred, titulo="Matriz de confusión"):
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation="nearest")
        ax.set_title(titulo)
        ax.set_xlabel("Predicho")
        ax.set_ylabel("Real")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, str(v), ha="center", va="center")
        fig.colorbar(im, ax=ax)
        return fig, ax
