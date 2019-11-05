import tensorflow as tf

IMG_SHAPE = 20
MODEL_DIR = "./models/"


def create_model():
    l0 = tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation=tf.nn.relu,
                                input_shape=(IMG_SHAPE, IMG_SHAPE, 1))
    p0 = tf.keras.layers.MaxPooling2D((2, 2), strides=2)

    l1 = tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation=tf.nn.relu)
    p1 = tf.keras.layers.MaxPooling2D((2, 2), strides=2)

    l2 = tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation=tf.nn.relu)
    p2 = tf.keras.layers.MaxPooling2D((2, 2), strides=2)

    do0 = tf.keras.layers.Dropout(0.5)

    fl1 = tf.keras.layers.Flatten()

    d0 = tf.keras.layers.Dense(128, activation=tf.nn.relu)

    d1 = tf.keras.layers.Dense(26, activation=tf.nn.softmax)

    model = tf.keras.Sequential([l0, p0, l1, p1, l2, p2, do0, fl1, d0, d1])

    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    print(model.summary())

    return model


def save_model(model):
    model.save(MODEL_DIR + "classifier.h5")
