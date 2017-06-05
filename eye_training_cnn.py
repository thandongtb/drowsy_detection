from __future__ import absolute_import
from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD
import pickle
import time

# Main
if __name__ == '__main__':
    pickle_files = ['open_eyes.pickle', 'closed_eyes.pickle']
    i = 0
    for pickle_file in pickle_files:
        with open(pickle_file, 'rb') as f:
            save = pickle.load(f)
            if i == 0:
                train_dataset = save['train_dataset']
                train_labels = save['train_labels']
                test_dataset = save['test_dataset']
                test_labels = save['test_labels']
            else:
                print("here")
                train_dataset = np.concatenate((train_dataset, save['train_dataset']))
                train_labels = np.concatenate((train_labels, save['train_labels']))
                test_dataset = np.concatenate((test_dataset, save['test_dataset']))
                test_labels = np.concatenate((test_labels, save['test_labels']))
            del save  # hint to help gc free up memory
        i += 1

    print('Training set', train_dataset.shape, train_labels.shape)
    print('Test set', test_dataset.shape, test_labels.shape)

    batch_size = 30
    nb_classes = 1
    nb_epoch = 12

    X_train = train_dataset
    # X_train = X_train.reshape((X_train.shape[0], X_train.shape[3]) + X_train.shape[1:3])
    Y_train = train_labels

    X_test = test_dataset
    # X_test = X_test.reshape((X_test.shape[0], X_test.shape[3]) + X_test.shape[1:3])
    Y_test = test_labels

    # print data shape
    print("{1} train samples, {4} channel{0}, {2}x{3}".format("" if X_train.shape[1] == 1 else "s", *X_train.shape))
    print("{1}  test samples, {4} channel{0}, {2}x{3}".format("" if X_test.shape[1] == 1 else "s", *X_test.shape))
    # input image dimensions
    _, img_channels, img_rows, img_cols = X_train.shape

    model = Sequential()
    # first	and	second Conv Layers with pooling
    model.add(Conv2D(32, (3, 3), padding='same',
                            input_shape=(img_channels, img_rows, img_cols)))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # FC layer
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    # Output layer. Define the class
    model.add(Dense(nb_classes))
    model.add(Activation('sigmoid'))

    # let's train the model using SGD + momentum (how original).
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

    model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, verbose=2, validation_data=(X_test, Y_test))

    score = model.evaluate(X_test, Y_test, verbose=0)

    print('Loss score:', score[0])
    print('Test accuracy:', score[1] * 100, '%')

    # Save model to file
    now = time.time()
    print("Save model to file json...")
    model_json = model.to_json()
    with open('trained_model/model_' + str(now) + '.json', "w") as json_file:
        json_file.write(model_json)

    print("Save weights to file...")
    model.save_weights('trained_model/weight_' + str(now) + '.h5', overwrite=True)