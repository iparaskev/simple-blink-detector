import csv
import numpy as np 
import keras 
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D,Flatten,Dense,Activation,Dropout,MaxPooling2D
from keras.activations import relu
from keras.optimizers import Adam

#we will use images 26x34x1 (1 is for grayscale images)
height = 26
width = 34
dims = 1

def readCsv(path):

	with open(path,'r') as f:
		#read the scv file with the dictionary format 
		reader = csv.DictReader(f)
		rows = list(reader)

	#imgs is a numpy array with all the images
	#tgs is a numpy array with the tags of the images
	imgs = np.empty((len(list(rows)),height,width, dims),dtype=np.uint8)
	tgs = np.empty((len(list(rows)),1))
		
	for row,i in zip(rows,range(len(rows))):
			
		#convert the list back to the image format
		img = row['image']
		img = img.strip('[').strip(']').split(', ')
		im = np.array(img,dtype=np.uint8)
		im = im.reshape((26,34))
		im = np.expand_dims(im, axis=2)
		imgs[i] = im

		#the tag for open is 1 and for close is 0
		tag = row['state']
		if tag == 'open':
			tgs[i] = 1
		else:
			tgs[i] = 0
	
	#shuffle the dataset
	index = np.random.permutation(imgs.shape[0])
	imgs = imgs[index]
	tgs = tgs[index]

	#return images and their respective tags
	return imgs,tgs	

#make the convolution neural network
def makeModel():
	model = Sequential()

	model.add(Conv2D(32, (3,3), padding = 'same', input_shape=(height,width,dims)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Conv2D(64, (2,2), padding= 'same'))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(128, (2,2), padding='same'))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Flatten())
	model.add(Dense(512))
	model.add(Activation('relu'))
	model.add(Dense(512))
	model.add(Activation('relu'))
	model.add(Dense(1))
	model.add(Activation('sigmoid'))

	
	model.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy',metrics=['accuracy'])

	return model

def main():

	xTrain ,yTrain = readCsv('dataset.csv')
	print (xTrain.shape[0])
	#scale the values of the images between 0 and 1
	xTrain = xTrain.astype('float32')
	xTrain /= 255

	model = makeModel()

	#do some data augmentation
	datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.2,
        height_shift_range=0.2,
        )
	datagen.fit(xTrain)

	#train the model
	model.fit_generator(datagen.flow(xTrain,yTrain,batch_size=32),
						steps_per_epoch=len(xTrain) / 32, epochs=50)
	
	#save the model
	model.save('blinkModel.hdf5')

if __name__ == '__main__':
	main()
