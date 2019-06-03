import keras

# for image processing
import numpy as np
from PIL import Image
from keras.preprocessing import image
from keras.applications.resnet50 import decode_predictions

# to build our own layers
from keras.layers import Input, Dense

from keras.models import Model  #functional API model
from keras.layers import Dense, Flatten
from keras.utils import to_categorical

# to plot things
import matplotlib.pyplot as plt

# loading the VGG16 already trained cnn, except for the top layers
vgg16_features = keras.applications.vgg16.VGG16(include_top = False, weights = 'imagenet')

# We define VGG16 as not trained
vgg16_features.trainable = False

# Load images
'''
bank = []
count = 0

images = glob.glob("cats-dogs-2000/*.jpg")
for image in images:
	with open(image, 'rb') as file:
		count = count + 1
		# Open image
		img = Image.open(file)
		# Resize to (224, 224)
		im_resized = img.resize((224, 224))
		# Convert to NumPy float array
		im_resized_np = np.asarray(im_resized, dtype = float)
		#resize
		im_resized_np = np.reshape(im_resized_np, (1, 224, 224, 3))
		bank.append(im_resized_np)

bank = np.reshape(bank, (count, 224, 224, 3))
'''

inputs = Input(shape=(224, 224, 3))
features_out = vgg16_features(inputs)
out_flat = Flatten()(features_out)
out_hidden1 = Dense(64, activation = 'relu')(inputs)
out_hidden2 = Dense(64, activation = 'relu')(out_hidden1)
predictions = Dense(1, activation = 'softmax')(out_hidden2)

cat_or_dog = Model(inputs = inputs, outputs = predictions)
cat_or_dog.summary()
