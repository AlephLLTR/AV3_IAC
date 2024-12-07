import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Funções Auxiliares
def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def sign(u):
    return 1 if u >= 0 else -1


def train_perceptron(X, Y, epochs, lr, train_size):
    N = X.shape[0]
    W = np.random.randn(X.shape[1])
    for _ in range(epochs):
        indexes = np.arange(N)
        np.random.shuffle(indexes)
        split = int(N * train_size)

        X_train, X_test = X[indexes[:split]], X[indexes[split:]]
        Y_train, Y_test = Y[indexes[:split]], Y[indexes[split:]]

        for x_t, y_t in zip(X_train, Y_train):
            u_t = np.dot(W, x_t)
            y_pred = sign(u_t)
            error = y_t - y_pred
            W += lr * error * x_t

    return W, X_test, Y_test


def predict(W, X):
    return np.array([sign(np.dot(W, x)) for x in X])


def calculate_metrics(predictions, Y):
    TP = np.sum((predictions == 1) & (Y == 1))
    TN = np.sum((predictions == -1) & (Y == -1))
    FP = np.sum((predictions == 1) & (Y == -1))
    FN = np.sum((predictions == -1) & (Y == 1))

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    sensitivity = TP / (TP + FN) if (TP + FN) != 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) != 0 else 0

    return accuracy, sensitivity, specificity, np.array([[TP, FN], [FP, TN]])


# Carregamento dos Dados
filepath = "Datasets/spiral.csv"
data = np.loadtxt(filepath, delimiter=",")
X_raw = data[:, :2]
Y = data[:, 2]

X = normalize(X_raw)

# Treinamento
epochs = 100
lr = 0.1
train_size = 0.8

W, X_test, Y_test = train_perceptron(X, Y, epochs, lr, train_size)

# Predições
predictions = predict(W, X_test)

# Métricas
accuracy, sensitivity, specificity, confusion = calculate_metrics(predictions, Y_test)

print(f"Acurácia: {accuracy:.2f}")
print(f"Sensibilidade: {sensitivity:.2f}")
print(f"Especificidade: {specificity:.2f}")

# Visualização
plt.figure(figsize=(8, 6))
sns.heatmap(
    confusion,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Positivo", "Negativo"],
    yticklabels=["Verdadeiro", "Falso"],
)
plt.xlabel("Predição")
plt.ylabel("Atual")
plt.title("Matriz de Confusão")
# plt.show()
