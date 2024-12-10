# Importações
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Definições
filepath = "Datasets/spiral.csv"
data = np.loadtxt(filepath, delimiter=",")
results = []
epochs = 500
p, N, C = 2, 2000, 2
lr = 0.1  # Taxa de Aprendizado
size = 0.8  # Tamanho da Amostra de Treinamento


# Função Sinal
def sign(u):
    return 1 if u >= 0 else -1


# Definições iniciais de X, W e Y
X = data[:, :2]  # Duas primeiras colunas como atributos
Y = data[:, 2]  # Terceira coluna como rótulos

# Normalização entre [-1, 1]
X = 2 * ((X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))) - 1

# Inicialização de pesos W_z
W_z = np.random.rand(X.shape[1])  # Dimensão igual ao número de características
X_axis = np.linspace(-1, 1)

# Pré-visualização do Gráfico (Comentado, caso necessário)
# plt.scatter(X[:, 0], X[:, 1])
# plt.show()

# Início do Treinamento
rounds = 50
general_accuracy = []
general_sensitivity = []
general_specificity = []
for round in range(rounds):
    for epoca in range(epochs):
        # Separando conjunto de amostras de treino e teste
        indexes = np.arange(N)
        np.random.shuffle(indexes)

        ind_train = indexes[: int(N * size)]
        ind_tests = indexes[int(N * size) :]

        X_train, X_test = X[ind_train], X[ind_tests]
        Y_train, Y_test = Y[ind_train], Y[ind_tests]

        # Atualização dos pesos com a regra de aprendizado
        for t in range(int(N * size)):
            x_t = X_train[t]
            d_t = Y_train[t]
            u_t = np.dot(W_z, x_t)  # Produto escalar
            e_t = d_t - u_t
            W_z += lr * e_t * x_t  # Atualização dos pesos

    # Cálculo da linha de decisão (caso o gráfico seja necessário)
    # x2 = -(W_z[0] * X_axis + W_z[2]) / W_z[1]
    # x2 = np.nan_to_num(x2)

    # Cálculo de métricas
    predictions = np.array([sign(np.dot(W_z, x)) for x in X_test])

    TP = np.sum((predictions == 1) & (Y_test.flatten() == 1))
    TN = np.sum((predictions == -1) & (Y_test.flatten() == -1))
    FP = np.sum((predictions == 1) & (Y_test.flatten() == -1))
    FN = np.sum((predictions == -1) & (Y_test.flatten() == 1))

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    sensitivity = TP / (TP + FN) if (TP + FN) != 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) != 0 else 0

    general_accuracy.append(accuracy)
    general_sensitivity.append(sensitivity)
    general_specificity.append(specificity)

    # Exibição das métricas
    # print("ADALINE: Rodada ", round)
    # print(f"Acurácia: {accuracy:.2f}")
    # print(f"Sensibilidade: {sensitivity:.2f}")
    # print(f"Especificidade: {specificity:.2f}")

    # Matriz de Confusão
    confusion = np.array([[TP, FN], [FP, TN]])

    # plt.pause(0.1)

print("ADALINE: Resultados Gerais")
print(f"Acurácia Média: {np.mean(general_accuracy):.2f}")
print(f"Acurácia Mínima: {np.min(general_accuracy):.2f}")
print(f"Acurácia Máxima: {np.max(general_accuracy):.2f}")
print(f"Acurácia Desvio: {np.std(general_accuracy):.2f}")

print(f"Sensibilidade Média: {np.mean(general_sensitivity):.2f}")
print(f"Sensibilidade Mínima: {np.min(general_sensitivity):.2f}")
print(f"Sensibilidade Máxima: {np.max(general_sensitivity):.2f}")
print(f"Sensibilidade Desvio: {np.std(general_sensitivity):.2f}")

print(f"Especificidade Média: {np.mean(general_specificity):.2f}")
print(f"Especificidade Mínima: {np.min(general_specificity):.2f}")
print(f"Especificidade Máxima: {np.max(general_specificity):.2f}")
print(f"Especificidade Desvio: {np.std(general_specificity):.2f}")

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
plt.show()

# TODO: Curva de Aprendizado
