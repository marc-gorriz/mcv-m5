# Keras imports
from keras.models import *
from keras.layers import *

IMAGE_ORDERING = 'channels_last'

def build_segnet(img_shape=(416, 608, 3), nclasses=8, l2_reg=0.,
               init='glorot_uniform', path_weights=None,
               load_pretrained=False, freeze_layers_from=None, basic=True):
    kernel = 3

    encoding_layers = [
        Convolution2D(64, kernel, border_mode='same', input_shape=img_shape),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(64, kernel, border_mode='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(),

        Convolution2D(128, kernel, kernel, border_mode='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(128, kernel, kernel, border_mode='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(),

        Convolution2D(256, kernel, kernel, border_mode='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(256, kernel, kernel, border_mode='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(256, kernel, kernel, border_mode='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(),

        Convolution2D(512, kernel, kernel, border_mode='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(512, kernel, kernel, border_mode='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(512, kernel, kernel, border_mode='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(),

        Convolution2D(512, kernel, kernel, border_mode='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(512, kernel, kernel, border_mode='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(512, kernel, kernel, border_mode='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(),
    ]

    model = Sequential()
    model.encoding_layers = encoding_layers

    for l in model.encoding_layers:
        model.add(l)
        print(l.input_shape, l.output_shape, l)

    decoding_layers = [
        UpSampling2D(),
        Convolution2D(512, kernel, kernel, border_mode='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(512, kernel, kernel, border_mode='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(512, kernel, kernel, border_mode='same'),
        BatchNormalization(),
        Activation('relu'),

        UpSampling2D(),
        Convolution2D(512, kernel, kernel, border_mode='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(512, kernel, kernel, border_mode='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(256, kernel, kernel, border_mode='same'),
        BatchNormalization(),
        Activation('relu'),

        UpSampling2D(),
        Convolution2D(256, kernel, kernel, border_mode='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(256, kernel, kernel, border_mode='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(128, kernel, kernel, border_mode='same'),
        BatchNormalization(),
        Activation('relu'),

        UpSampling2D(),
        Convolution2D(128, kernel, kernel, border_mode='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(64, kernel, kernel, border_mode='same'),
        BatchNormalization(),
        Activation('relu'),

        UpSampling2D(),
        Convolution2D(64, kernel, kernel, border_mode='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(nclasses, 1, 1, border_mode='valid'),
        BatchNormalization(),
    ]

    model.decoding_layers = decoding_layers
    for l in model.decoding_layers:
        model.add(l)

        model.add(Reshape((nclasses, img_shape[0] * img_shape[1])))
        model.add(Permute((2, 1)))
        model.add(Activation('softmax'))

    # Freeze some layers
    if freeze_layers_from is not None:
        freeze_layers(model, freeze_layers_from)

    return model


# Freeze layers for finetunning
def freeze_layers(model, freeze_layers_from):
    # Freeze the VGG part only
    if freeze_layers_from == 'base_model':
        print('   Freezing base model layers')
        freeze_layers_from = 23

    # Show layers (Debug pruposes)
    for i, layer in enumerate(model.layers):
        print(i, layer.name)
    print('   Freezing from layer 0 to ' + str(freeze_layers_from))

    # Freeze layers
    for layer in model.layers[:freeze_layers_from]:
        layer.trainable = False
    for layer in model.layers[freeze_layers_from:]:
        layer.trainable = True
