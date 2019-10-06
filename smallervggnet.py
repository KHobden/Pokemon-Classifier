#VGGNet Network (Copied from PyImageSearch)
#Kieran Hobden
#26-Sep-'19

C:\Users\kiera\AppData\Local\Programs\Python\Python36\Scripts\cnn-keras\cnn-keras

from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K

class SmallerVGGNet:
	@staticmethod
	def build(width, height, depth, classes):
		#Initialise the model and use channels_last for use in TensorFlow
		model = Sequential()
		inputShape = (height, width, depth)
		chanDim = -1

		#If we need to use channels_first, update the selection
		if K.image_data_format() == "channels_first":
			inputShape = (depth, height, width)
			chanDim = 1

		#CONV => RELU => POOL (filter size 32, spatial dimensions reduced from 96x96 to 32x32)
		model.add(Conv2D(32, (3, 3), padding="same", input_shape=inputShape))
		model.add(Activation("relu")) 
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(3, 3)))
		model.add(Dropout(0.25))

		#(CONV => RELU) * 2 => POOL (filter size 64, spatial dimensions reduced from 96x96 to 48x48)
		model.add(Conv2D(64, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(Conv2D(64, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))

		#(CONV => RELU) * 2 => POOL (filter size 128, spatial dimensions reduced from 96x96 to 32x32)
		model.add(Conv2D(128, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(Conv2D(128, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))

		#FC => RELU (dropout rate of 50%)
		model.add(Flatten())
		model.add(Dense(1024))
		model.add(Activation("relu"))
		model.add(BatchNormalization())
		model.add(Dropout(0.5))

		#Softmax classifier
		model.add(Dense(classes))
		model.add(Activation("softmax"))

		#Return the constructed network architecture
		return model
