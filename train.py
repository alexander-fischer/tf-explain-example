import tensorflow as tf
import numpy as np
from tf_explain.callbacks.integrated_gradients import IntegratedGradientsCallback

import logging

from trainer.data_loader import load_data, create_datagenerator
from trainer.model import create_model, save_model

BATCH_SIZE = 32
EPOCHS = 20
OUTPUT_DIR = "./output/"


def train_and_evaluate():
    x_train, x_test, y_train, y_test = load_data()
    train_examples = x_train.shape[0]
    test_examples = x_test.shape[0]

    train_data_gen, val_data_gen = create_datagenerator(x_train, x_test, y_train, y_test)

    # Take one example out of test data set
    x_example = np.array(x_test[0:1])
    y_example = np.array(y_test[0:1])

    model = create_model()

    # Setup integrated gradient
    # For more possibilities check https://github.com/sicara/tf-explain
    explain_callback = IntegratedGradientsCallback(
        validation_data=(x_example, y_example),
        class_index=0,
        n_steps=20,
        output_dir=OUTPUT_DIR,
    )
    callbacks = [explain_callback]

    model.fit_generator(
        train_data_gen,
        steps_per_epoch=(train_examples // BATCH_SIZE),
        epochs=EPOCHS,
        validation_data=val_data_gen,
        validation_steps=(test_examples // BATCH_SIZE),
        callbacks=callbacks
    )

    save_model(model)


if __name__ == "__main__":
    logger = tf.get_logger()
    logger.setLevel(logging.ERROR)

    print(tf.__version__)

    train_and_evaluate()
