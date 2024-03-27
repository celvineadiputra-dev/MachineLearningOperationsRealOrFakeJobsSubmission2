# Project Requirement

## Menggunakan TensorFlow Extended (TFX) untuk Membuat Machine Learning Pipeline

Dalam project ini, Anda akan membangun sebuah machine learning pipeline sederhana dengan menggunakan TensorFlow Extended (TFX).
Pipeline ini diharapkan mampu menyelesaikan tugas JOB POSTING PREDICTION dengan memuat seluruh komponen yang dibutuhkan, seperti:

- [X] ExampleGen
- [X] StatisticGen
- [X] SchemaGen
- [X] ExampleValidator
- [X] Transform
- [X] Trainer
- [X] Resolver
- [X] Evaluator
- [X] Pusher
- [ ] Menjalankan Sistem Machine Learning Menggunakan Komputasi Cloud
- [ ] Memantau Sistem Machine Learning Menggunakan Prometheus
- [X] Memanfaatkan komponen Tuner untuk menjalankan proses hyperparameter tuning secara otomatis.
- [X] Menerapkan prinsip clean code dalam membuat machine learning pipeline.  
- [ ] Menambahkan sebuah berkas notebook untuk menguji dan melakukan prediction request ke sistem machine learning yang telah dijalankan di cloud.
- [ ] Menyinkronkan Prometheus dengan Grafana untuk membuat dashboard monitoring yang lebih menarik.

## Formatter

```
pip install autopep8
```

```
autopep8 --in-place --aggressive --aggressive <FILE_NAME>
```

## Pylint

```
pip install pylint
```

```
pylint modules/*.py
```