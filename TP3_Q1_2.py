# TESTING NORMALAZATION WITH DROPOUT's IMPORTANCE

import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

# to add convolution relayer
from keras.layers.convolutional import Conv2D
# to add pooling layers
from keras.layers.convolutional import MaxPooling2D
# to flatten layers and to also add dropout and dense
from keras.layers import Flatten, Dropout, Dense

# to plot things
import matplotlib.pyplot as plt

# to import minst data 
from keras.datasets import mnist

# Reseau convolution simple 
def cnn_keras_model():
	model = Sequential()
	# 32, 3x3 convolution filters
	model.add(Conv2D(32, kernel_size = (3, 3), activation = 'relu', input_shape = input_shape))
	# 64, 3x3 convolution filters
	model.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu'))
	model.add(MaxPooling2D(pool_size = (2,3)))

	# WE TAKE THE 25% DROPOUT OFF

	# convert data from 2D to 1D
	model.add(Flatten()) 
	# dense layer of 128 neurons
	model.add(Dense(128, activation = 'relu'))

	# WE TAKE THE 50% DROPOUT OFF

	model.add(Dense(num_classes, activation = 'softmax'))

	# Adam optimizer
	model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
	return model


# fixing image dimensions
img_rows = 28
img_cols = 28

# reading and preparing the data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
	X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
	X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
	input_shape = (1,img_rows, img_cols)
else :
	X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
	X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
	input_shape = (img_rows, img_cols, 1)

# normalizing data
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# encode outputs
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
num_classes = y_test.shape[1]
print("num_calasses = ", num_classes)

model = cnn_keras_model()

# training on 12 époques
hist = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=12, verbose=2)

# Evaluating the module
scores = model.evaluate(X_test, y_test, verbose=0)

# check precision
print('---->Accuracy: {}% \n---->Error: {}'.format(scores[1], 1 - scores[1]))

# PLOTTING TRAINING EVOLUTION
xvals = range(12)		# 12 data points (because we do 12 epoques)
plt.clf()				# Clear figure
# plot boreth training and validation accuracy on the same figure
plt.plot(xvals, hist.history["acc"], label = "Traning accuracy")
plt.plot(xvals, hist.history["val_acc"], label = "Validation accuracy")
plt.legend() # display legend
plt.show() # show the figure

# output
#	---->Accuracy: 0.9876%		& 		---->Error: 0.012399999999999967
# On remaruqe une grande difference par rapport a la premier partie ou on fesait la normalisation

