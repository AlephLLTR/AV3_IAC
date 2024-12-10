import numpy as np, os, cv2, matplotlib.pyplot as plt
from tqdm import tqdm


def carregar_imagens(dim=40, pasta="Q2/RecFac", classes=20):
    X, Y = [], []
    for i, p in enumerate(os.listdir(pasta)):
        for img in os.listdir(os.path.join(pasta, p)):
            img_resized = cv2.resize(
                cv2.imread(os.path.join(pasta, p, img), cv2.IMREAD_GRAYSCALE),
                (dim, dim),
            )
            X.append(img_resized.flatten() / 255.0)
            y = np.full(classes, -1)
            y[i] = 1
            Y.append(y)
    X, Y = np.array(X).T, np.array(Y).T
    return (X - X.mean(axis=1, keepdims=True)) / (
        X.std(axis=1, keepdims=True) + 1e-8
    ), Y


class RNA:
    def __init__(self, inp, out, lr=0.001, epochs=50):
        self.w, self.b, self.lr, self.ep = (
            np.random.randn(out, inp) * 0.01,
            np.zeros((out, 1)),
            lr,
            epochs,
        )

    def treinar(self, X, Y):
        for _ in tqdm(range(self.ep), desc="Treinando RNA"):
            for i in range(X.shape[1]):
                x, y = X[:, i : i + 1], Y[:, i : i + 1]
                err = y - np.where(np.dot(self.w, x) + self.b >= 0, 1, -1)
                self.w += self.lr * np.dot(err, x.T)
                self.b += self.lr * err

    def prever(self, X):
        z = np.dot(self.w, X) + self.b
        Y = -np.ones(z.shape)
        idx = np.argmax(z, axis=0)
        for i, ix in enumerate(idx):
            Y[ix, i] = 1
        return Y


class ADALINE(RNA):
    def treinar(self, X, Y):
        for _ in tqdm(range(self.ep), desc="Treinando ADALINE"):
            output = np.dot(self.w, X) + self.b  # Ativação Linear
            error = Y - output  # Cálculo do erro
            self.w += self.lr * np.dot(error, X.T)  # Ajuste dos pesos
            self.b += self.lr * np.sum(error, axis=1, keepdims=True)  # Ajuste do bias

    def prever(self, X):
        z = np.dot(self.w, X) + self.b
        return (z == z.max(axis=0)).astype(int) * 2 - 1  # Retorna -1 ou 1 (bipolar)


def calcular_metricas(Y_true, Y_pred):
    tp, fn = (Y_true == 1) & (Y_pred == 1), (Y_true == 1) & (Y_pred == -1)
    tn, fp = (Y_true == -1) & (Y_pred == -1), (Y_true == -1) & (Y_pred == 1)
    return (
        (tp.sum() + tn.sum()) / max(tp.sum() + tn.sum() + fp.sum() + fn.sum(), 1),
        tp.sum() / max(tp.sum() + fn.sum(), 1),
        tn.sum() / max(tn.sum() + fp.sum(), 1),
    )


def matriz_confusao(Y_true, Y_pred, classes):
    cm = np.zeros((classes, classes), dtype=int)
    true_labels, pred_labels = np.argmax(Y_true, axis=0), np.argmax(Y_pred, axis=0)
    for t, p in zip(true_labels, pred_labels):
        cm[t, p] += 1
    return cm


def plot_matriz_confusao(cm, title="Matriz de Confusão"):
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(title, fontsize=16, fontweight="bold")
    plt.colorbar(label="Frequência")
    plt.xlabel("Classe Predita", fontsize=14)
    plt.ylabel("Classe Verdadeira", fontsize=14)
    plt.xticks(range(len(cm)), range(len(cm)), fontsize=12)
    plt.yticks(range(len(cm)), range(len(cm)), fontsize=12)
    for i in range(len(cm)):
        for j in range(len(cm)):
            plt.text(
                j,
                i,
                f"{cm[i, j]}",
                ha="center",
                va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
                fontsize=12,
            )
    plt.tight_layout()
    plt.show()


def monte_carlo(modelo, X, Y, rodadas=50):
    resultados, confusoes = [], []
    for _ in tqdm(range(rodadas), desc="Monte Carlo"):
        idx = np.random.permutation(X.shape[1])
        split = int(0.8 * X.shape[1])
        Xt, Yt, Xv, Yv = (
            X[:, idx[:split]],
            Y[:, idx[:split]],
            X[:, idx[split:]],
            Y[:, idx[split:]],
        )
        novo_modelo = type(modelo)(X.shape[0], Y.shape[0], modelo.lr, modelo.ep)
        novo_modelo.treinar(Xt, Yt)
        Y_pred = novo_modelo.prever(Xv)
        resultados.append(calcular_metricas(Yv, Y_pred))
        confusoes.append(matriz_confusao(Yv, Y_pred, Y.shape[0]))
    return int(np.mean(resultados, axis=0)), int(np.mean(confusoes, axis=0))


if __name__ == "__main__":
    X, Y = carregar_imagens()

    # Perceptron
    perceptron = RNA(X.shape[0], Y.shape[0], lr=0.001, epochs=50)
    res_perceptron, cm_perceptron = monte_carlo(perceptron, X, Y)
    print("\nPerceptron:")
    print(f"Média de Acurácia: {res_perceptron[0]:.4f}")
    print(f"Sensibilidade Média: {res_perceptron[1]:.4f}")
    print(f"Especificidade Média: {res_perceptron[2]:.4f}")
    plot_matriz_confusao(cm_perceptron, title="Matriz de Confusão - Perceptron")

    # ADALINE
    adaline = ADALINE(X.shape[0], Y.shape[0], lr=0.0005, epochs=100)
    res_adaline, cm_adaline = monte_carlo(adaline, X, Y)
    print("\nADALINE:")
    print(f"Média de Acurácia: {res_adaline[0]:.4f}")
    print(f"Sensibilidade Média: {res_adaline[1]:.4f}")
    print(f"Especificidade Média: {res_adaline[2]:.4f}")
    plot_matriz_confusao(cm_adaline, title="Matriz de Confusão - ADALINE")
