"""
TUNER
"""
from typing import NamedTuple, Dict, Text, Any
import tensorflow as tf
import keras_tuner as kt
import tensorflow_transform as tft
from tensorflow.keras import layers
from keras_tuner.engine import base_tuner
from tfx.components.trainer.fn_args_utils import FnArgs

LABEL_KEY = "fraudulent"
FEATURE_KEY = "full_description"

NUM_EPOCHS = 5

TunerFnResult = NamedTuple("TunerFnResult", [
    ("tuner", base_tuner.BaseTuner),
    ("fit_kwargs", Dict[Text, Any]),
])

early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor="val_binary_accuracy",
    mode="max",
    verbose=1,
    patience=10,
)


def transformed_name(key):
    """
    A function that takes a key and transforms it by appending '_xf' to it.

    Parameters:
    key (str): The key to be transformed.

    Returns:
    str: The transformed key with '_xf' appended.
    """
    return f"{key}_xf"


def gzip_reader_fn(filenames):
    """
    A function that generates a TFRecordDataset from the given filenames with GZIP compression.

    Parameters:
    filenames (list): A list of filenames to read the TFRecordDataset from.

    Returns:
    TFRecordDataset: A TFRecordDataset object with GZIP compression.
    """
    return tf.data.TFRecordDataset(filenames, compression_type="GZIP")


def input_fn(file_pattern, tf_transform_output, num_epochs, batch_size=64):
    """
    Generate a batched dataset from files matching the file pattern.

    Args:
        file_pattern (str): Pattern of files to read data from.
        tf_transform_output (TFTransformOutput): Output of the TF Transform process.
        num_epochs (int): Number of epochs to iterate over the dataset.
        batch_size (int, optional): Size of batches to create, default is 64.

    Returns:
        tf.data.Dataset: A batched dataset for training or evaluation.
    """
    transform_feature_spec = (
        tf_transform_output.transformed_feature_spec().copy()
    )

    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transform_feature_spec,
        reader=gzip_reader_fn,
        num_epochs=num_epochs,
        label_key=transformed_name(LABEL_KEY),
    )

    return dataset


def model_builder(hp, vectorized_layer):
    """
    Builds a Keras model for binary classification.

    Args:
        hp (HyperParameters): The hyperparameters for the model.
        vectorized_layer (tf.keras.layers.TextVectorization): The vectorized layer for text data.

    Returns:
        tf.keras.Model: The compiled Keras model.

    Hyperparameters:
        - num_hidden_layers (int): The number of hidden layers in the model. Must be either 1 or 2.
        - embed_dims (int): The dimension of the embedding layer. Must be between 16 and 128,
        with a step of 32.
        - lstm_units (int): The number of units in the LSTM layer. Must be between 32 and 128,
        with a step of 32.
        - dense_units (int): The number of units in the dense layer. Must be between 32 and 256,
        with a step of 32.
        - dropout_rate (float): The dropout rate for the dropout layer. Must be between 0.1 and 0.5,
        with a step of 0.1.
        - learning_rate (float): The learning rate for the optimizer. Must be one of 1e-2, 1e-3, or
        1e-4.

    Model Architecture:
        - Input layer: Takes a string input of shape (1,)
        with the name transformed_name(FEATURE_KEY).
        - Vectorized layer: Applies the vectorized layer to the input.
        - Embedding layer: Maps the vectorized input to an embedding of dimension embed_dims.
        - Bidirectional LSTM layer: Applies a bidirectional LSTM to the embedding.
        - Hidden layers: Applies dense and dropout layers num_hidden_layers times.
        - Output layer: Applies a dense layer with a
        sigmoid activation function to produce a binary output.

    Model Compilation:
        - Compiles the model with the Adam optimizer, binary cross entropy loss,
        and binary accuracy metric.

    """
    num_hidden_layers = hp.Choice(
        "num_hidden_layers", values=[1, 2]
    )
    embed_dims = hp.Int(
        "embed_dims", min_value=16, max_value=128, step=32
    )
    lstm_units = hp.Int(
        "lstm_units", min_value=32, max_value=128, step=32
    )
    dense_units = hp.Int(
        "dense_units", min_value=32, max_value=256, step=32
    )
    dropout_rate = hp.Float(
        "dropout_rate", min_value=0.1, max_value=0.5, step=0.1
    )
    learning_rate = hp.Choice(
        "learning_rate", values=[1e-2, 1e-3, 1e-4]
    )

    inputs = tf.keras.Input(
        shape=(1,), name=transformed_name(FEATURE_KEY), dtype=tf.string
    )

    x = vectorized_layer(inputs)
    x = layers.Embedding(input_dim=5000, output_dim=embed_dims)(x)
    x = layers.Bidirectional(layers.LSTM(lstm_units))(x)

    for _ in range(num_hidden_layers):
        x = layers.Dense(dense_units, activation=tf.nn.relu)(x)
        x = layers.Dropout(dropout_rate)(x)

    outputs = layers.Dense(1, activation=tf.nn.sigmoid)(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=["binary_accuracy"],
    )

    return model


def tuner_fn(fn_args: FnArgs):
    """
    Initialize and tune a model using hyperband optimization.

    Args:
        fn_args (FnArgs): An object containing all the necessary arguments for the function.

    Returns:
        TunerFnResult: A named tuple containing the tuned model and fitting arguments.
    """

    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    train_set = input_fn(
        fn_args.train_files[0], tf_transform_output, NUM_EPOCHS
    )
    eval_set = input_fn(
        fn_args.eval_files[0], tf_transform_output, NUM_EPOCHS
    )

    vectorizer_dataset = train_set.map(
        lambda f, l: f[transformed_name(FEATURE_KEY)]
    )

    vectorizer_layer = layers.TextVectorization(
        max_tokens=5000,
        output_mode="int",
        output_sequence_length=500,
    )
    vectorizer_layer.adapt(vectorizer_dataset)

    tuner = kt.Hyperband(
        hypermodel=lambda hp: model_builder(hp, vectorizer_layer),
        objective=kt.Objective('binary_accuracy', direction='max'),
        max_epochs=NUM_EPOCHS,
        factor=3,
        directory=fn_args.working_dir,
        project_name="kt_hyperband",
    )

    return TunerFnResult(
        tuner=tuner,
        fit_kwargs={
            "callbacks": [early_stopping_callback],
            "x": train_set,
            "validation_data": eval_set,
            "steps_per_epoch": fn_args.train_steps,
            "validation_steps": fn_args.eval_steps,
        },
    )
