import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
import matplotlib.pyplot as plt

np.random.seed(0)

x1 = np.random.randn(100, 2) + np.array([0, 10])
x2 = np.random.randn(100, 2) + np.array([5, 5])
x3 = np.random.randn(100, 2) + np.array([10, 0])
y1 = np.array([[1, 0, 0] for i in range(100)])
y2 = np.array([[0, 1, 0] for i in range(100)])
y3 = np.array([[0, 0, 1] for i in range(100)])

for p in x1:
    plt.scatter(p[0], p[1], s=5, c="red")

for p in x2:
    plt.scatter(p[0], p[1], s=5, c="orange")

for p in x3:
    plt.scatter(p[0], p[1], s=5, c="blue")

x = np.concatenate((x1, x2, x3), axis=0)
y = np.concatenate((y1, y2, y3), axis=0)

model = Sequential()
model.add(Dense(3, input_dim=2))
model.add(Activation("softmax"))
model.compile(loss="categorical_crossentropy", optimizer=SGD(lr=0.1))
model.fit(x, y, batch_size=50, epochs=100)


def f(x, d, c1, c2):
    a = -((d[0][0, c1] - d[0][0, c2]) / (d[0][1, c1] - d[0][1, c2]))
    b = -((d[1][c1] - d[1][c2]) / (d[0][1, c1] - d[0][1, c2]))
    return a * x + b


xmin, xmax = min(x[:, 0]), max(x[:, 0])
xs = np.linspace(xmin, xmax, 100)
d = model.layers[0].get_weights()

y = [f(x, d, 0, 1) for x in xs]
plt.plot(xs, y, 'r')

y = [f(x, d, 1, 2) for x in xs]
plt.plot(xs, y, 'g')

plt.show()
