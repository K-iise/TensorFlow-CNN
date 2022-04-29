import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# keras의 데이터셋의 패션 MNIST 데이터를 불러와서 학습용, 테스트 데이터로 구분.
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) =\
fashion_mnist.load_data()


mnist_lbl = ['T-shirt', 'Trouser','Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle_boot']
# lbl 리스트 생성
labels = train_labels[:4] # train_labels[0] ~ train_labels[3]의 값의 리스트
for i in labels: 
    print('{} : {}'.format(i, mnist_lbl[i]))

