import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import Adam
from keras.initializers import he_normal
from keras.callbacks import EarlyStopping
from keras.backend import tensorflow_backend as backend
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

np.set_printoptions(threshold=np.nan)

mnist = datasets.fetch_mldata("MNIST original", data_home=".")

N = 30000

indices = np.random.permutation(len(mnist.data))[:N]
x = mnist.data[indices]
x = x / x.max()
x = x - x.mean(axis=1).reshape(len(x), 1)
y = mnist.target[indices].astype(int)
e = np.eye(10)
y = e[y]
x_train, x_test, y_train, y_test = train_test_split(x, y)
print("data:", N)
print("test:", len(x_test))

# 28*28サイズの画像->200ノードの隠れ層->200ノードの隠れ層->200ノードの隠れ層->10個の数字
input_size = 28*28
hidden_size = [200, 200, 200]
output_size = 10

model = Sequential()
for i, input_dim in enumerate(([input_size] + hidden_size)[:-1]):
    model.add(
        Dense(
            hidden_size[i],
            input_dim=input_dim,
            kernel_initializer=he_normal()))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))

model.add(Dense(output_size, kernel_initializer=he_normal()))
model.add(Activation("softmax"))

model.compile(
    optimizer="Adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"])

early_stopping = EarlyStopping(patience=10)

history = model.fit(
    x_train,
    y_train,
    batch_size=200,
    epochs=200,
    verbose=2,
    callbacks=[early_stopping],
    validation_split=0.25)

print(model.evaluate(x_test, y_test))

backend.clear_session()

fig = plt.figure()
e = len(history.history["acc"])
print("stop epoch:", e)

ax_val_acc = fig.add_subplot(1, 1, 1)
ln_val_acc = ax_val_acc.plot(
    range(e), history.history["val_acc"], label="val_acc", color="r")
ax_val_loss = ax_val_acc.twinx()
ln_val_loss = ax_val_loss.plot(
    range(e), history.history["val_loss"], label="val_loss", color="b")

lns = ln_val_acc + ln_val_loss
labels = [l.get_label() for l in lns]
ax_val_acc.legend(lns, labels, loc=0)

plt.xlabel("epoch")
ax_val_acc.set_ylabel("val_acc")
ax_val_loss.set_ylabel("val_loss")
plt.show()
