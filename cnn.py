'''
Using a convolutional neural network to classify images into two classes:
dog: 1
cat: 0
'''

from keras import layers
from keras import models
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
import matplotlib.pyplot as plt

# Instantiating a convnet for classification of dogs vs cats
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# Configuring the model for training
model.compile(loss='binary_crossentropy', metrics=['accuracy'],
              optimizer=optimizers.RMSprop(lr=1e-4))

# Data Preprocessing
# Using keras built in utilities to convert image files
# into batches of preprocessed tensors
train_datagen = ImageDataGenerator(rescale=1./255) # Rescales images by 1/255
test_datagen = ImageDataGenerator(rescale=1./255)

# Directories where the training set and validation set are located
train_dir = '/home/tevinachong/Code/deep_learning_with_python/dogs_vs_cats/minimized_data/train'
validation_dir = '/home/tevinachong/Code/deep_learning_with_python/dogs_vs_cats/minimized_data/validation'

'''
Creates batches of size 20, each containing tensor version of the images in the dataset
the shape of the batches will be (20, 150, 150, 3)
'''
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150), # Resizes the images to 150 x 150
    batch_size=20,
    class_mode='binary' # Because we use binary_crossentropy loss, we need binary labels
)

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary'
)

# Fitting the model using a batch generator
history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=50
)

# Saving the model
model.save('../models/cats_and_dogs_small_1.h5')

# Displaying curves of loss and accuracy during training
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training Accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.show()


