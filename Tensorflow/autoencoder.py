from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
import PIL.Image as pil
from tensorflow.keras.layers import Dense, Flatten, Reshape
from tensorflow.keras import Model

mnist = tf.keras.datasets.mnist
#data label (output) ignore output now

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

# print(x_train[0])
num_pick = 1
original_images = np.zeros((num_pick,28,28))
original_test = np.zeros((num_pick,28,28))

for i in range(num_pick):
    index = np.random.choice(range(60000))
    original_images[i] = x_train[index]
    index = np.random.choice(range(10000))
    original_test[i]= x_test[index]
    

#vector for decoding
latent_dim = 64 

class Autoencoder(Model):
  def __init__(self, latent_dim):
    super(Autoencoder, self).__init__()
    self.latent_dim = latent_dim   
    self.encoder = tf.keras.Sequential([
      Flatten(),
      Dense(latent_dim, activation='relu'),
    ])
    self.decoder = tf.keras.Sequential([
      Dense(784, activation='sigmoid'),
      Reshape((28, 28))
    ])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

autoencoder = Autoencoder(latent_dim)

autoencoder.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())

autoencoder.fit(x_train,x_train,
                epochs=10,
                shuffle=True,
                validation_data=(x_test, x_test))

# encoded_imgs = autoencoder.encoder(x_test).numpy()
for i in original_images:
    img = pil.fromarray(np.int8(i*255.) , 'L')
    img.show()

encoded_images = autoencoder.encoder(original_images).numpy()
decoded_images= autoencoder.decoder(encoded_images).numpy()

for i in decoded_images:
    print(np.max(i))
    img = pil.fromarray(np.int8(i*255.) , 'L')
    img.show()