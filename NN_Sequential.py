import os

import sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import cv2 as cv

if __name__ == "__main__":
    physical_devices = tf.config.experimental.list_physical_devices("GPU")
    if physical_devices:
        print(physical_devices)
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# data visualization
# print(f"x_train {x_train.shape} Y_train{y_train.shape}")
# img=x_train[10]
# lbl=y_train[10]
# print("label",lbl)
# cv.imshow("image",img)
# cv.waitKey(0)
# cv.destroyAllWindows()
"""
Flatten the  images to feed into the nn, 
basically converting the 28*28 tensor into one long tensor 
and also convert to  black and white

"""
x_train = x_train.reshape(-1, 28 * 28).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28 * 28).astype("float32") / 255.0

"""
Sequential ApI
 Very convenient ,but not very flexible
 Means you can use it when you have one input map to specifically one output
"""

# model = keras.Sequential(
#     [
#         layers.Input(28 * 28),
#         layers.Dense(512, activation="relu"),  # fully connected
#         layers.Dense(512, activation="relu"),
#         layers.Dense(10),
#     ]
# )

model=keras.Sequential()
model.add(keras.Input(shape=(28*28)))
model.add(layers.Dense(512,activation="relu"))
model.add(layers.Dense(512,activation="relu"))
model.add(layers.Dense(256,activation="relu"))
model.add(layers.Dense(10,activation="softmax"))

"""debugging in deep learning"""
# model=keras.Model(inputs=model.inputs,outputs=[layer.output for layer in model.layers])
# features=model.predict(x_train)
# for f in features:
#     print(f.shape)
# sys.exit()
#funtional api
# inputs=keras.Input(shape=(28*28),name="input")
# x=layers.Dense(512,activation='relu',name="first_layer")(inputs)
# x=layers.Dense(256,activation='relu',name="second_layer")(x)
# outputs=layers.Dense(10,activation="softmax")(x)
# model=keras.Model(inputs=inputs,outputs=outputs)


model.summary()

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(
        from_logits=False
    ),  # this uses softmax at the last layer
    optimizer=keras.optimizers.SGD(learning_rate=0.01),
    # model.compile(loss='categorical_crossentropy', optimizer=opt)
    metrics=["accuracy"],
)

model.fit(x_train, y_train, batch_size=16, epochs=20, verbose=2)
model.evaluate(x_test, y_test, batch_size=16, verbose=2)
