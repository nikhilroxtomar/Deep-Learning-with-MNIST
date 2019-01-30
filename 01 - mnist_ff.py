## Importing Packages
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
import tensorflow.keras.layers as KL

## Dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train/255.0, x_test/255.0

## Model
inputs = KL.Input(shape=(28, 28))                      #(?, 28, 28)
l = KL.Flatten()(inputs)                               #(?, 784)
l = KL.Dense(512, activation=tf.nn.relu)(l)            #(?, 512)
outputs = KL.Dense(10, activation=tf.nn.softmax)(l)    #(?, 10) -> (?, 1)

model = tf.keras.models.Model(inputs, outputs)
model.summary()
model.compile(optimizer="adam",
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"])
model.fit(x_train, y_train, epochs=5)
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test Loss: {0} - Test Acc: {1}".format(test_loss, test_acc))
