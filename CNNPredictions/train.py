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

for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            
            NAME = 'BN-{}-conv-{}-node-{}-dens-{}'.format(conv_layer, layer_size, dense_layer, int(time.time()))  # model name with timestamp
            print(NAME) 
            
            tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))
            callbacks = [tensorboard]
            
            model = Sequential()
            
            # first layer
            model.add(Conv2D(layer_size, (3,3), padding="same", activation="relu", input_shape=inputShape))
            model.add(BatchNormalization())
            model.add(MaxPooling2D(pool_size=(3,3)))
            
            # sets up additional # of conv layers
            for _ in range(conv_layer - 1):
                layer_size *= 2
                model.add(Conv2D(layer_size, (3,3), padding="same", activation="relu"))
                model.add(BatchNormalization())
                model.add(Conv2D(layer_size, (3,3), padding="same", activation="relu"))
                model.add(BatchNormalization())
                model.add(MaxPooling2D(pool_size=(2,2)))
                model.add(Dropout(0.25))
            
            model.add(Flatten())
            
            layer_size *= 4 # to get the dense layer to be 8X of last output size
            
            # sets up # of dense layers
            for _ in range(dense_layer):
                model.add(Dense(layer_size, activation='relu'))
                model.add(BatchNormalization())
                model.add(Dropout(0.5))
            
            # output layer
            model.add(Dense(1))
            model.add(Activation('sigmoid'))
            
            opt = Adam(lr=0.001)
            model.compile(loss='binary_crossentropy', 
                          optimizer=opt,
                          metrics=['accuracy'])

            model.fit_generator(generator=train_generator,
                                steps_per_epoch=(train_generator.n // train_generator.batch_size),
                                callbacks = callbacks,
                                validation_data=val_generator,
                                validation_steps=(val_generator.n // val_generator.batch_size),
                                epochs=10,
                                use_multiprocessing=False,
                                workers=4)

pickle_out = open('model_10_epoch.pickle', 'wb')
pickle.dump(model, pickle_out)
pickle_out.close()

