import keras

# for image processing
import numpy as np
from PIL import Image
from keras.preprocessing import image
from keras.applications.resnet50 import decode_predictions

# to plot things
import matplotlib.pyplot as plt

# to parse through a file of pictures
import glob

# loading the VGG16 already trained cnn
vgg16 = keras.applications.vgg16.VGG16(include_top = True, weights = 'imagenet')


bank = []
count = 0

images = glob.glob("image/*.jpg")
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

print('number of samples :', count, "\n")
print('check size data array', np.array(bank).shape, "\n")


predicts = vgg16.predict(bank)
print('Predicted', decode_predictions(predicts, top = 3)[0])

# output:
# Predicted [('n02119789', 'kit_fox', 0.34652036), ('n01877812', 'wallaby', 0.17589845), ('n02120505', 'grey_fox', 0.13314462)]
# on conclut le bon fonctionnemnet de ce reseau, il arrive a bien reconnaitre l animal