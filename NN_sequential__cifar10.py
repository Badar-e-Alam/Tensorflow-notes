import os
from pickletools import optimize
os.environ['TF_CPP_MIN_LOG_LEVEL']="2"

import tensorflow  as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10 


if __name__=="__main__":
    physical_devices = tf.config.experimental.list_physical_devices("GPU")
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

(x_train,y_train),(x_test,y_test)=cifar10.load_data()
x_train=x_train.astype("float32")/255.0
x_test=x_test.astype("float32")/255.0

model=keras.Sequential()
model.add(layers.Input(shape=(32,32,3)))
model.add(layers.Conv2D(32,3,padding="valid",activation="relu"))
model.add(layers.MaxPooling2D())
model.add(layers.Conv2D(64,3,activation="relu"))
model.add(layers.MaxPooling2D())
model.add(layers.Conv2D(128,3,activation="relu"))
model.add(layers.Flatten())
model.add(layers.Dense(64,activation="relu"))
model.add(layers.Dense(10))
model.summary()

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(learning_rate=0.01),
    metrics=['accuracy'],
)
model.fit(x_train, y_train, batch_size=16, epochs=5, verbose=2)
model.evaluate(x_test, y_test, batch_size=16, verbose=2)