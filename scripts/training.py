import tensorflow as tf
from tensorflow.python.keras.utils.np_utils import to_categorical
from tensorflow.keras.layers import Input, Dense

def mnist_model():
    optimizer_ = "adam"
    loss_ = "sparse_categorical_crossentropy"
    metrics_ = ["accuracy"]
    model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape = (28,28)), tf.keras.layers.Dense(128,activation = 'relu'), tf.keras.layers.Dropout(0.2),tf.keras.layers.Dense(10,activation = 'softmax')])
    model.compile(optimizer = optimizer_, loss = loss_, metrics = metrics_)
    return model

if __name__ == "__main__":
    mnist = tf.keras.datasets.mnist
    (x_train,y_train),(x_test,y_test) = mnist.load_data()


    EPOCHS = 10
    classifier = mnist_model()
    history = classifier.fit(x = x_train, y = y_train, epochs = EPOCHS)

    classifier.save("./opt/ml/processing/output")