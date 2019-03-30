# Example: python retrain.py -i=my_last_model.h5 -o=my_new_model -lr=0.0005 -e=10


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i","--inputModel", help="file path to an existing model, ie: myModel.h5", type=str)
parser.add_argument("-o", "--outputModelName", help="name of output model, ie: myNewModel", type=str)
parser.add_argument("-lr","--learnRate", help="learning rate to be used, default=0.001", type=float)
parser.add_argument("-e","--epochs", help="number of epochs to run, default=1", type=int)


args = parser.parse_args()
inputModel = args.inputModel
NAME = args.outputModelName
learnRate = args.learnRate
epochs = args.epochs


if (inputModel != None):
	print('input model: ', inputModel)
else:
	print('input model not specified. Usage: --inputModel=myModelFile.h5')
	exit()

if (NAME == None):
	NAME = inputModel.split('.')[0] + '-2'
	
print('New model: ', NAME)

if (learnRate == None):
	learnRate = 0.001

print('Learning rate: ', learnRate)

if (epochs == None):
	epochs = 1

print('Number of epochs: ', epochs)




import os
os.environ['KERAS_BACKEND']='plaidml.keras.backend'

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, ZeroPadding2D, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard
from keras_preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
import time
import pickle
from keras.models import load_model

conv_layers = [5]      # number of conv layers
layer_sizes = [32]     # number of nodes in a layer
dense_layers = [2]     # number of dense layers

pickle_in = open('../Dataset/df_all.pickle', 'rb')
df_train, df_test = pickle.load(pickle_in)

# The Keras ImageDataGenerator uses string type data label
df_train['gender'] = df_train.gender.astype(str)
df_test['gender'] = df_test.gender.astype(str)

image_reshape_size = 120
input_image_root_dir = '../Dataset/imdb_crop/' # Don't forget the ending slash

from keras import backend as K
K.set_image_data_format('channels_last')
batch_size = 64
inputShape = (image_reshape_size, image_reshape_size, 1)

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.1)

train_generator = datagen.flow_from_dataframe(dataframe=df_train,
                                            directory=input_image_root_dir,
                                            x_col="path", y_col="gender",
                                            subset="training",
                                            class_mode="binary",
                                            color_mode="grayscale",
                                            target_size=(image_reshape_size,image_reshape_size),
                                            batch_size=32,
                                            seed=1,
                                            shuffle=True)

val_generator = datagen.flow_from_dataframe(dataframe=df_train,
                                            directory=input_image_root_dir,
                                            x_col="path", y_col="gender",
                                            subset="validation",
                                            class_mode="binary",
                                            color_mode="grayscale",
                                            target_size=(image_reshape_size,image_reshape_size),
                                            batch_size=32,
                                            seed=1,
                                            shuffle=True)

test_generator = datagen.flow_from_dataframe(dataframe=df_test, 
                                            directory=input_image_root_dir, 
                                            x_col="path", y_col=None, 
                                            class_mode=None, 
                                            color_mode="grayscale",
                                            target_size=(image_reshape_size,image_reshape_size),
                                            batch_size=1,
                                            shuffle=False)



tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))
callbacks = [tensorboard]

opt = Adam(lr=learnRate)
model.compile(loss='binary_crossentropy', 
              optimizer=opt,
              metrics=['accuracy'])

model.fit_generator(generator=train_generator,
                    steps_per_epoch=(train_generator.n // train_generator.batch_size),
                    callbacks = callbacks,
                    validation_data=val_generator,
                    validation_steps=(val_generator.n // val_generator.batch_size),
                    epochs=epochs,
                    use_multiprocessing=False,
                    workers=2)

filepath = NAME + '.h5'
model.save(filepath)