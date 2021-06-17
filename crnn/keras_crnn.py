from keras import backend as K
from keras.layers import Dense, Dropout, Flatten, Input, Permute, Reshape, Softmax
from keras.layers.advanced_activations import ELU
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.layers.recurrent import GRU, LSTM
from keras.models import Model
from keras.utils.data_utils import get_file

def createCRNNModel(
    input_shape,
    num_classes=10,
    normalized=False,
    conv_num=2,
    conv_filter=64,
    conv_kernel_size=3,
    conv_stride=3):
    melgram_input = Input(shape=input_shape)
    if K.image_data_format() == 'channels_last':
        time_axis, freq_axis, channel_axis = 1, 2, 3
    else:
        channel_axis, freq_axis, time_axis = 1, 2, 3
    
    # Building the model
    x = melgram_input if normalized else BatchNormalization(axis=freq_axis, name='bn_0_freq')(melgram_input)

    for conv_idx in range(conv_num):
        x = Convolution2D(
            conv_filter,
            conv_kernel_size,
            conv_stride,
            padding='same',
            name='conv' + str(conv_idx + 1))(x)
        # x = BatchNormalization(axis=channel_axis, name='bn1')(x)
        # x = ELU()(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool' + str(conv_idx + 1))(x)
        x = Dropout(0.2, name='dropout' + str(conv_idx + 1))(x)

    # x = Flatten()(x)
    # x = Dense(num_classes, activation='softmax')(x)

    # reshaping
    if K.image_data_format() == 'channels_first':
        x = Permute((3, 1, 2))(x)
    x = Reshape((-1, 128))(x)
    # RNN block 1, 2, output
    x = LSTM(num_classes, return_sequences=False, name='lstm1', activation='softmax')(x)
    x = Dropout(0.1)(x)
    
    # Create model
    model = Model(melgram_input, x)
    return model
