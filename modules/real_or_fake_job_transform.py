import tensorflow as tf


def real_or_fake_job_transform(args):
    def transformed_name(key):
        return key + "_xf"

    def preprocessing_fn(inputs):
        outputs = {}

        outputs[transformed_name(FEATURE_KEY)] = tf.strings.lower(
            inputs[args['FEATURE_KEY']])

        outputs[transformed_name(LABEL_KEY)] = tf.cast(
            inputs[args['LABEL_KEY']], tf.int64)

        return outputs

    return (
        transformed_name,
        preprocessing_fn
    )
