# Submission 2: JOB POSTING PREDICTION

Nama: Celvine Adi Putra

Username dicoding: CelvineAdiPutra

<img src="https://dicoding-web-img.sgp1.cdn.digitaloceanspaces.com/original/submission-rating-badge/rating-default-5.png"/>

| | Deskripsi |
| ----------- | ----------- |
| Dataset | [Real / Fake Job Posting Prediction](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction?resource=download) |
| Masalah | Penipuan lowongan kerja semakin marak terjadi. Lowongan palsu dibuat untuk menipu pelamar dan mengambil keuntungan pribadi, seperti uang atau informasi pribadi. Hal ini dapat menyebabkan kerugian bagi pelamar, baik secara finansial maupun emosional. |
| Solusi machine learning | Machine learning dapat digunakan untuk memprediksi apakah suatu lowongan kerja adalah asli atau palsu. Model machine learning dapat dilatih dengan data yang berisi contoh lowongan kerja asli dan palsu. Dengan demikian, model dapat mempelajari pola dan ciri-ciri yang membedakan kedua jenis lowongan tersebut. |
| Metode pengolahan | Metode pengelolahan data yang digunakan yaitu tokenisasi sebagai fitur input, data awal yang berupa text akan dilakukan data cleansing, dan pemilihan fitur yang akan digunakan. Proses ini memiliki tujuan agar data dapat di pahami oleh model |
| Arsitektur model | Model ini dibangun menggunakan layer TextVectorization, sebagi layer yang berfungsi untuk memproses input string kedalam bentuk angka, dan layer embedding bertugas untuk mengukur kedekatan atau kesamaan dari setiap kata untuk mengetahui  kata tersebut merupakan kata negatif atau kata positif. |
| Metrik evaluasi | Mertrik evaluasi yang di gunakan pada model ini yaiut TP, TN, FP, dan FN |
| Performa model | Model yang telah dibuat mendapatkan hasil yang cukup baik dengan akurasi 98% dalam memprediksi job posting. |
| Opsi deployment | Project Machine Learning ini di deploy di layanan Cloudeka |
| Web app | [Serving Model URL](http://103.190.215.94:8501/v1/models/real-or-fake-jobs-detection-model/) |
| Monitoring | Project ini di monitoring dengan prometheus menggunakan grafana|
