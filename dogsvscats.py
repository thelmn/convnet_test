#%% [Markdown]
# Convnet lessons using kaggle dogs vs cats dataset
# [https://www.kaggle.com/c/dogs-vs-cats/data]


#%%
import os, shutil
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt

#%%
# subsample dogsvscats train set to 2000 training, 1000 validation, 1000 test
#%%
subsample_base = './data/dogsvscats/subsample'
train_dir = os.path.join(subsample_base, 'train')
validation_dir = os.path.join(subsample_base, 'validation')
test_dir = os.path.join(subsample_base, 'test')

def subsample():
    original_dir = './data/dogsvscats/train'

    if not os.path.exists(subsample_base): 
        os.mkdir(subsample_base)

    train_cats_dir = os.path.join(train_dir, 'cats')
    train_dogs_dir = os.path.join(train_dir, 'dogs')

    validation_cats_dir = os.path.join(validation_dir, 'cats')
    validation_dogs_dir = os.path.join(validation_dir, 'dogs')

    test_cats_dir = os.path.join(test_dir, 'cats')
    test_dogs_dir = os.path.join(test_dir, 'dogs')

    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
    if not os.path.exists(validation_dir):
        os.mkdir(validation_dir)
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)

    if not os.path.exists(train_cats_dir):
        os.mkdir(train_cats_dir)
    if not os.path.exists(train_dogs_dir):
        os.mkdir(train_dogs_dir)

    if not os.path.exists(validation_cats_dir):
        os.mkdir(validation_cats_dir)
    if not os.path.exists(validation_dogs_dir):
        os.mkdir(validation_dogs_dir)

    if not os.path.exists(test_cats_dir):
        os.mkdir(test_cats_dir)
    if not os.path.exists(test_dogs_dir):
        os.mkdir(test_dogs_dir)

    train_cats_size = len(os.listdir(train_cats_dir))
    print('total training cat images:', train_cats_size)
    if train_cats_size == 1000:
        return

    fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
    for fname in fnames:
        src = os.path.join(original_dir, fname)
        dst = os.path.join(train_cats_dir, fname)
        shutil.copyfile(src, dst)
    
    fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
    for fname in fnames:
        src = os.path.join(original_dir, fname)
        dst = os.path.join(validation_cats_dir, fname)
        shutil.copyfile(src, dst)

    fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]
    for fname in fnames:
        src = os.path.join(original_dir, fname)
        dst = os.path.join(test_cats_dir, fname)
        shutil.copyfile(src, dst)

    fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
    for fname in fnames:
        src = os.path.join(original_dir, fname)
        dst = os.path.join(train_dogs_dir, fname)
        shutil.copyfile(src, dst)
    
    fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
    for fname in fnames:
        src = os.path.join(original_dir, fname)
        dst = os.path.join(validation_dogs_dir, fname)
        shutil.copyfile(src, dst)

    fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
    for fname in fnames:
        src = os.path.join(original_dir, fname)
        dst = os.path.join(test_dogs_dir, fname)
        shutil.copyfile(src, dst)

#%%
subsample()

#%%
# build the simple convnet
# - Dropout layer at dropout rate of 0.5 to reduce overfitting.
#   Dropout randomly zeros half of the outputs of the previous layer;
#   equivalent to randomly shifting the active nodes to prevent any
#   'conspiracies'.
#%%
model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Conv2D(128, (3,3), activation='relu'))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Conv2D(128, (3,3), activation='relu'))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
print(model.summary())
#%%
model.compile(
    optimizer=optimizers.RMSprop(lr=1e-4),
    loss='binary_crossentropy',
    metrics=['acc']
)

#%% 
# Image generator. lazily loads images from disk 
# - Data augmentation by generating more data from original set by 
#   rotating, shifting, sheering, scaling, flipping.
#   Reduces overfitting when the available training dataset is small
#%%
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
    )
train_data = datagen.flow_from_directory(
    directory=train_dir,
    target_size=(150,150),
    batch_size=20,
    class_mode='binary'
    )

validation_data = ImageDataGenerator(rescale=1./255).flow_from_directory(
    directory=validation_dir,
    target_size=(150,150),
    batch_size=20,
    class_mode='binary'
)

#%%
# connect model to generator and train
#%%
epochs = 10
history = model.fit_generator(
    train_data,
    steps_per_epoch=100,
    epochs=epochs,
    validation_data=validation_data,
    validation_steps=50
)
#%%
# save the model
#%%
model.save('dogs_vs_cats_model_1.h5')

#%% 
# plot accuracy and loss
#%%
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epoch_range = range(1, epochs+1)

plt.plot(epoch_range, acc, 'bo', label='Training acc')
plt.plot(epoch_range, val_acc, 'b', label='Validation acc')
plt.title('accuracy')
plt.legend()

plt.figure()

plt.plot(epoch_range, loss, 'bo', label='Training loss')
plt.plot(epoch_range, val_loss, 'b', label='Validation loss')
plt.title('loss')
plt.legend()

plt.show()

#%%
