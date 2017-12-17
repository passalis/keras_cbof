from __future__ import print_function
from dataset import load_mnist, resize_images
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, GlobalMaxPool2D
from keras.layers import Conv2D, MaxPooling2D
from cbof import BoF_Pooling, initialize_bof_layers
from spp import SpatialPyramidPooling



def build_model(pool_type='max', n_output_filters=32, n_codewords=32):
    if pool_type == 'max':
        input_size = 28
    else:
        input_size = None

    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(input_size, input_size, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(n_output_filters, (3, 3), activation='relu'))

    if pool_type == 'max':
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
    elif pool_type == 'gmp':
        model.add(GlobalMaxPool2D())
    elif pool_type == 'spp':
        model.add(SpatialPyramidPooling([1, 2]))
    elif pool_type == 'bof':
        model.add(BoF_Pooling(n_codewords, spatial_level=0))
    elif pool_type == 'spatial_bof':
        model.add(BoF_Pooling(n_codewords, spatial_level=1))
    else:
        assert Flatten

    model.add(Dropout(0.2))
    model.add(Dense(512, activation='elu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(lr=0.001),
                  metrics=['accuracy'])

    return model


def evaluate_model(pool_type, n_filters=128, n_codewords=0):
    print("Evaluating model: ", pool_type)
    model = build_model(pool_type, n_output_filters=n_filters, n_codewords=n_codewords)
    print(model.summary())
    initialize_bof_layers(model, x_train)

    model.fit(x_train, y_train, batch_size=128, epochs=50, verbose=2, validation_data=(x_test, y_test))
    acc = 100 * model.evaluate(x_test, y_test, verbose=0)[1]
    print('Test error:', 100 - acc)

    if pool_type != 'max':
        acc1 = 100 * model.evaluate(resize_images(x_test, scale=0.8), y_test, verbose=0)[1]
        print("Test error (scale=0.8): ", 100 - acc1)


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = load_mnist()

    # Baseline model
    evaluate_model('max', n_filters=64)
    evaluate_model('max', n_filters=32)
    evaluate_model('max', n_filters=4)

    # Global Max Pooling
    evaluate_model('gmp', n_filters=16)
    evaluate_model('gmp', n_filters=24)
    evaluate_model('gmp', n_filters=64)
    evaluate_model('gmp', n_filters=128)

    # SPP
    evaluate_model('spp', n_filters=8)
    evaluate_model('spp', n_filters=16)
    evaluate_model('spp', n_filters=32)
    evaluate_model('spp', n_filters=64)

    # CBoF
    evaluate_model('bof', n_filters=32, n_codewords=16)
    evaluate_model('bof', n_filters=32, n_codewords=64)
    evaluate_model('bof', n_filters=32, n_codewords=128)

    # Spatial CBoF
    evaluate_model('spatial_bof', n_filters=32, n_codewords=8)
    evaluate_model('spatial_bof', n_filters=32, n_codewords=16)
    evaluate_model('spatial_bof', n_filters=64, n_codewords=32)

