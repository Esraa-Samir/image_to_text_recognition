def create_model(init_mode='uniform', activation_mode='linear', optimizer_mode="adam", activation_mode_conv='linear'):
    model = Sequential()

    model.add(ZeroPadding2D((6, 4), input_shape=(6, 3, 3)))
    model.add(Convolution2D(32, 3, 3, activation=activation_mode_conv))
    print model.output_shape
    model.add(Convolution2D(32, 3, 3, activation=activation_mode_conv))
    print model.output_shape
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 1)))
    print model.output_shape
    model.add(Convolution2D(64, 3, 3, activation=activation_mode_conv))
    print model.output_shape
    model.add(Convolution2D(64, 3, 3, activation=activation_mode_conv))
    print model.output_shape
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 1)))
    model.add(Flatten())
    print model.output_shape
    model.add(Dense(output_dim=32, input_dim=64, init=init_mode, activation=activation_mode))
    model.add(Dense(output_dim=13, input_dim=50, init=init_mode, activation=activation_mode))
    model.add(Dense(output_dim=1, input_dim=13, init=init_mode, activation=activation_mode))
    model.add(Dense(output_dim=1, init=init_mode, activation=activation_mode))
    # print model.summary()
    model.compile(loss='mean_squared_error', optimizer=optimizer_mode)

    return model
