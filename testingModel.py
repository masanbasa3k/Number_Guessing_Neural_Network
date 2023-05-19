import install_requirements

import keras
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_test = tf.keras.utils.normalize(X_test, axis=1)

for test in range(len(X_test)):
    for row in range(28):
        for x in range(28):
            if X_test[test][row][x] != 0:
                X_test[test][row][x] = 1


model = tf.keras.models.load_model('awesome_legendary_number_reader.model')
predictions = model.predict(X_test[:10])

count = 0
for x in range(len(predictions)):
    guess = (np.argmax(predictions[x]))
    actual = y_test[x]

    print(f'Prediction number is : {guess}')
    print(f'Real number is : {actual}')

    if guess != actual:
        count+=1

    plt.imshow(X_test[x], cmap=plt.cm.binary)
    plt.show()
    
print(f'Model  got {count} wrong guesses in {len(predictions)}')
print(f'{str(100 - ((count/len(predictions))*100))}% correct')