import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mp


mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) =\
mnist.load_data()

model = keras.Sequential([  # keras.Squential() 클래스는 입력층, 은닉층, 출력층이 존재하는 신경망 모델을 생성.
    keras.layers.Flatten(input_shape=(28,28)), # Flatten() 클래스는 2차원 이미지을 1차원 값으로 바꿔줌. 매개변수는 입력의 크기 28 * 28 = 784가 입력 노드의 수.  입력층
    keras.layers.Dense(256, activation='relu'), # Dense() 클래스는 매개 변수로 몇 개의 출력으로 연결할지 정하는 값을 받음. 256개의 노드 구성 은닉층
    keras.layers.Dense(10, activation='softmax') #마지막의 Dense() 클래스는 입력 받은 784개의 값이 신경망을 통과하여 10개 중 하나의 범주로 분류함.
])
# activation은 활성화 함수로 relu는 음수를 0으로 만드는 함수 
# softmax 함수는 인풋값들을 넣으면, 그 값들을 모두 0과 1사이의 값으로 정규화하고 확률처럼 모든 아웃풋 값을 더했을 때 1이 총합이라는 특징을 갖는 함수

# 인공신경망의 구조.
# 입력층(input Layer)은 주어진 이미지의 픽셀값을 받아들인다.
# 은닉층(Hidden Layer)은 앞선 층의 출력을 입력으로 사용하여 연산 하고 그 결과를 뒤에 있는 입력으로 전달.
# 그리고 마지막으로 출력층(Output Layer)는 인공신경망 전체의 출력을 생성한다.  


model.compile(optimizer='adam', #최적화 기법으로 adam을 선택
              loss ='sparse_categorical_crossentropy', # 손실함수로 다중 분류 손실 함수를 사용
             metrics =['accuracy']) # metrics는 모델의 성능을 평가하는데 사용하는 함수. accuracy 정확도를 평가.
model.fit(train_images, train_labels, epochs=5) # fit() 메소드는 학습을 시작하게 함 
                                                # epochs는 학습 횟수. 

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\n테스트 정확도:',test_acc)

randIdx = np.random.randint(0,6000)
plt.imshow(test_images[randIdx],cmap = 'Greys')
plt.show()
pred = model.predict(test_images[randIdx][np.newaxis, :, :])

print(pred.argmax())
