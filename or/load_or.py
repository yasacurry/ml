import numpy as np
from keras.models import Sequential, load_model
import h5py
import os
import sys

x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [1]])

model = load_model(sys.argv[1])

classes = model.predict_classes(x, batch_size=1)
prob = model.predict_proba(x, batch_size=1)

print("targets == predict_classes?")
print(y == classes)
print("predict_proba")
print(prob)