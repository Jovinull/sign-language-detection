from keras import layers, models, regularizers
import config

def create_model():
    model = models.Sequential()

    # Camadas convolucionais dinamicamente de acordo com config
    for i in range(config.conv_layers):
        filters = config.filters[i]
        if i == 0:
            model.add(layers.Conv2D(filters, (config.kernel_size, config.kernel_size), activation=config.activation,
                                    input_shape=(config.IMG_SIZE, config.IMG_SIZE, 3),
                                    kernel_regularizer=regularizers.l2(config.l2_reg)))
        else:
            model.add(layers.Conv2D(filters, (config.kernel_size, config.kernel_size), activation=config.activation,
                                    kernel_regularizer=regularizers.l2(config.l2_reg)))
        model.add(layers.BatchNormalization())  # Adicionando BatchNormalization
        model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.GlobalAveragePooling2D())  # Global pooling ao invés de flatten

    # Camada densa e saída
    model.add(layers.Dense(config.dense_units, activation=config.activation,
                           kernel_regularizer=regularizers.l2(config.l2_reg)))
    model.add(layers.Dropout(config.dropout_rate))
    model.add(layers.Dense(21, activation='softmax'))  # Ajuste o número de unidades de saída conforme necessário

    # Compilando o modelo
    model.compile(optimizer=config.optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model
