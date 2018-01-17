import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
import h5py
import os

file_name = os.path.splitext(os.path.basename(__file__))
model_dir = os.path.join(os.path.dirname(__file__), file_name[0])
if os.path.exists(model_dir) is False:
    os.mkdir(model_dir)

np.random.seed(0)

x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [1]])

model = Sequential()
model.add(Dense(1, input_dim=2))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy', optimizer=SGD(lr=0.1))

model_path = model_dir + "/model.epoch_{epoch:03d}-loss{loss:.4f}.hdf5"
checkpoint_callback = ModelCheckpoint(filepath=model_path)

model.fit(x, y, batch_size=1, epochs=200, callbacks=[checkpoint_callback])

classes = model.predict_classes(x, batch_size=1)
prob = model.predict_proba(x, batch_size=1)

print("targets == predict_classes?")
print(y == classes)
print("predict_proba")
print(prob)