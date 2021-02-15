from keras import layers
from keras import models
from keras.preprocessing.image import ImageDataGenerator
import os
from keras import optimizers 
from keras import losses

train_dir = "C:\\Users\\Tadeas\\Downloads\\scenesapp\\seg_train\\seg_train"

val_dir = "C:\\Users\\Tadeas\\Downloads\\valdata\\seg_test\\seg_test"

m = models.Sequential()

m.add(layers.Conv2D(32, (3, 3), activation= "relu", input_shape= (224, 224, 3)))
m.add(layers.MaxPooling2D((2, 2)))
m.add(layers.Conv2D(64, (3, 3), activation= "relu"))
m.add(layers.MaxPooling2D((2, 2)))
m.add(layers.Conv2D(128, (3, 3), activation= "relu"))
m.add(layers.MaxPooling2D((2, 2)))
m.add(layers.Conv2D(128, (3, 3), activation= "relu"))
m.add(layers.MaxPooling2D((2, 2)))
m.add(layers.Conv2D(128, (3, 3), activation= "relu"))
m.add(layers.MaxPooling2D((2, 2)))
m.add(layers.Flatten())
m.add(layers.Dropout(0.5))
m.add(layers.Dense(512, activation= "relu"))
m.add(layers.Dense(6, activation = "softmax"))

trdatagen = ImageDataGenerator(rescale= 1./255, rotation_range= 40, width_shift_range= 0.2, height_shift_range= 0.2, shear_range= 0.2, zoom_range= 0.2, horizontal_flip= True)

valdatagen = ImageDataGenerator(rescale = 1./255)

train_generator = trdatagen.flow_from_directory(train_dir, target_size=(224, 224), batch_size=32, class_mode="categorical")

val_generator = valdatagen.flow_from_directory(val_dir, target_size=(224, 224), batch_size=32, class_mode="categorical")

m.compile(optimizer= optimizers.Adam(lr= 0.001), loss = "categorical_crossentropy", metrics=["acc"])

m.fit_generator(train_generator, steps_per_epoch=100, epochs=100, validation_data=val_generator, validation_steps=50)

m.save("Scenes.h5")