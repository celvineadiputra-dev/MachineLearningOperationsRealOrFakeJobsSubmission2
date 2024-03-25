"""
TRANSFORM
"""
import tensorflow as tf
import tensorflow_transform as tft

LABEL_KEY = "fraudulent"
FEATURE_KEY = "full_description"


def transformed_name(key):
    """
    A function that takes a key and transforms it by appending '_xf' to it.

    Parameters:
    key (any): The key to be transformed.

    Returns:
    str: The transformed key with '_xf' appended to it.
    """
    return key + "_xf"


def preprocessing_fn(inputs):
    """
    A function that preprocesses inputs by transforming feature and label keys.

    Parameters:
    inputs (dict): A dictionary containing feature and label keys.

    Returns:
    dict: A dictionary with transformed feature and label keys.
    """
    outputs = {transformed_name(FEATURE_KEY): tf.strings.lower(
        inputs[FEATURE_KEY]), transformed_name(LABEL_KEY): tf.cast(
        inputs[LABEL_KEY], tf.int64)}

    return outputs
