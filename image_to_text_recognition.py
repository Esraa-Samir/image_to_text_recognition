import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation
import os
import pickle
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix
from keras.models import load_model
from sklearn.metrics import classification_report, precision_recall_fscore_support
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV


def build_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(32, 32, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=26, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def load_data():
    train_data = ImageDataGenerator(rescale=1. / 255,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True)
    test_data = ImageDataGenerator(rescale=1. / 255)
    train_generator = train_data.flow_from_directory(
        directory='Data',
        target_size=(32, 32),
        batch_size=100,
        class_mode='categorical'
    )
    test_generator = test_data.flow_from_directory(
        directory='Testing',
        target_size=(32, 32),
        batch_size=100,
        class_mode='categorical')

    X, Y = train_generator.next()

    return train_generator, test_generator, X, Y

def train_model(train_generator, test_generator):
    checkpoint = ModelCheckpoint('.', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    history = model.fit(train_generator,
                        steps_per_epoch=16,
                        epochs=150,
                        callbacks=callbacks_list,
                        validation_data=test_generator,
                        validation_steps=16)

    return history

def evaluate_model(model, test_generator):
    scores = model.evaluate(test_generator, verbose=0)
    Y_pred = model.predict(test_generator)
    y_pred = np.argmax(Y_pred, axis=1)
    cm = confusion_matrix(test_generator.classes, y_pred)
    precision, recall, fscore, _ = precision_recall_fscore_support(test_generator.classes, y_pred, average='macro')

    print("Model Accuracy: " + "%.2f" % scores[1])
    print("Model Loss: " + "%.2f" % scores[0])
    print("Model Precision: " + "%.2f" % precision)
    print("Model Recall: " + "%.2f" % recall)
    print("Model fscore: " + "%.2f" % fscore)

def hyperparameters_tunning(x, y):
    model = KerasRegressor(build_fn=build_model, verbose=10)
    batch_size_list = [10, 20, 40, 60, 80, 100]
    epochs = [10, 50, 100, 150]
    # epochs= [2,5]
    param_grid = dict(batch_size=batch_size_list, nb_epoch=epochs)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1)
    grid_result = grid.fit(x,y)

    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))


model = build_model()
train_generator, test_generator, X, Y = load_data()
# hyperparameters_tunning(X, Y)
history = train_model(train_generator, test_generator)
evaluate_model(model, test_generator)
