
import os
import csv
import cv2
import numpy as np

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from keras.models import Sequential
from keras.layers import Dense, Flatten, Lambda
from keras.layers import Convolution2D, MaxPooling2D, Cropping2D, Dropout
from keras.callbacks import ModelCheckpoint
from keras.callbacks import Callback

# Get Data
samples = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

# Split data into train and test sets
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# Data Dir
data_path = './data'

# Cropping Variables
ty = 40
dy = -20

# Generator
def generator(samples, batch_size=32, colour_space='RGB'):

    # Get number of samples
    num_samples = len(samples)

    # Loop forever so the generator never terminates
    while True:

        # Shuffle samples
        shuffle(samples)

        # Iterate over samples
        for offset in range(0, num_samples, batch_size):

            # Get batches
            batch_samples = samples[offset:offset+batch_size]

            # Arrays to store data and labels
            images = []
            angles = []

            # Iterate over batches
            for batch_sample in batch_samples:

                # Get Images Path
                center_name = data_path + '/IMG/' + batch_sample[0].split('/')[-1].strip()
                left_name = data_path + '/IMG/' + batch_sample[1].split('/')[-1].strip()
                right_name = data_path + '/IMG/'+ batch_sample[1].split('/')[-1].strip()

                # Read images from center, left and right cameras
                center_image = cv2.imread(center_name)
                left_image = cv2.imread(left_name)
                right_image = cv2.imread(right_name)

                # Handle Colour Spaces Diferences
                if colour_space == 'RGB':
                    colour_conv = cv2.COLOR_BGR2RGB
                else:
                    colour_conv = cv2.COLOR_BGR2GRAY

                # Convert Colour Space
                center_image = cv2.cvtColor(center_image, colour_conv)
                left_image = cv2.cvtColor(left_image, colour_conv)
                right_image = cv2.cvtColor(right_image, colour_conv)

                # Reshape if not RGB to handle Numpy Error
                if colour_space != 'RGB':
                    center_image = np.reshape(center_image, (center_image.shape[0], center_image.shape[1], 1))
                    left_image = np.reshape(left_image, (center_image.shape[0], center_image.shape[1], 1))
                    right_image = np.reshape(right_image, (center_image.shape[0], center_image.shape[1], 1))

                # Crop Images - see only road section
                center_image = center_image[ty:dy,:, :]
                left_image = left_image[ty:dy,:, :]
                right_image = right_image[ty:dy,:, :]

                # Augment Images
                center_image_flip = cv2.flip(center_image, 1)
                left_image_flip = cv2.flip(left_image, 1)
                right_image_flip = cv2.flip(right_image, 1)

                # Reshape Augmented Images if not RGB to handle Numpy Error
                if colour_space != 'RGB':
                    center_image_flip = np.reshape(center_image_flip, (center_image_flip.shape[0], center_image_flip.shape[1], 1))
                    left_image_flip = np.reshape(left_image_flip, (center_image_flip.shape[0], center_image_flip.shape[1], 1))
                    right_image_flip = np.reshape(right_image_flip, (center_image_flip.shape[0], center_image_flip.shape[1], 1))


                # Get Measurements
                center_angle = float(batch_sample[3])

                # Create adjusted steering measurements for the side camera images
                correction = 0.1 # this is a parameter to tune
                left_angle = center_angle + correction
                right_angle = center_angle - correction

                # Augment Measurements
                center_angle_flip = center_angle * -1.0
                left_angle_flip = left_angle * -1.0
                right_angle_flip = right_angle * -1.0

                # Resize Images for Input
                center_image = cv2.resize(center_image, (200, 66), interpolation=cv2.INTER_AREA)
                left_image = cv2.resize(left_image, (200, 66), interpolation=cv2.INTER_AREA)
                right_image = cv2.resize(right_image, (200, 66), interpolation=cv2.INTER_AREA)
                center_image_flip = cv2.resize(center_image_flip, (200, 66), interpolation=cv2.INTER_AREA)
                left_image_flip = cv2.resize(left_image_flip, (200, 66), interpolation=cv2.INTER_AREA)
                right_image_flip = cv2.resize(right_image_flip, (200, 66), interpolation=cv2.INTER_AREA)

                # Append Images
                images.append(center_image)
                angles.append(center_angle)
                images.append(left_image)
                angles.append(left_angle)
                images.append(right_image)
                angles.append(right_angle)

                images.append(center_image_flip)
                angles.append(center_angle_flip)
                images.append(left_image_flip)
                angles.append(left_angle_flip)
                images.append(right_image_flip)
                angles.append(right_angle_flip)

            # Convert to Numpy arrays
            X_train = np.array(images)
            y_train = np.array(angles)

            # Shuffle and Yeld
            yield shuffle(X_train, y_train)


# Get training and validation generators
train_generator = generator(train_samples, batch_size=32, colour_space='RGB')
validation_generator = generator(validation_samples, batch_size=32, colour_space='RGB')

# Dropout Value
keep_prob = 0.2

# Build Model
model = Sequential()
# Normalization Layer
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(66, 200, 3)))
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Dropout(keep_prob))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Dropout(keep_prob))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Dropout(keep_prob))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Dropout(keep_prob))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Dropout(keep_prob))

model.add(Flatten())

model.add(Dense(100))
model.add(Dropout(keep_prob))
model.add(Dense(50))
model.add(Dropout(keep_prob))
model.add(Dense(10))
model.add(Dropout(keep_prob))
model.add(Dense(1))

# Load Previous Weights for Transfer Learning
model.load_weights("./models/previous_weights.h5")
# Compile model with Adam
model.compile(loss='mse', optimizer='adam',  metrics=['accuracy'])

# Check Point for saveing best model
# filepath="m-{epoch:02d}-{val_loss:.2f}.hdf5"
# checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
# callbacks_list = [checkpoint]

class SaveEachEpoch(Callback):
    def __init__(self, model):
        self.model = model
        self.epoch = 0

    def on_epoch_begin(self, epoch, logs={}):
        self.model.save('up-{}'.format(self.epoch))
        self.epoch += 1

callbacks_list = [SaveEachEpoch(model)]

# Samples Params
n_cameras = 3
n_augmentations = 2

# Fit Model
model.fit_generator(train_generator,
                     samples_per_epoch=len(train_samples) * n_cameras * n_augmentations,
                     validation_data=validation_generator,
                     nb_val_samples=len(validation_samples) * n_cameras * n_augmentations,
                     nb_epoch=17,
                     callbacks=callbacks_list)

# Save Model
model.save('model.h5')
