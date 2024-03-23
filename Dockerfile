FROM tensorflow/serving:latest

COPY ./serving_model_dir /models
ENV MODEL_NAME=real-or-fake-jobs-detection-model