from layer import *
from keras.layers import *

def meca_block(input_feature, k_size=3):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    channel = input_feature._keras_shape[channel_axis]

    shared_layer_one = Conv1D(filters=1,kernel_size=k_size,strides=1,kernel_initializer='he_normal',use_bias=False,padding="same")


    avg_pool = GlobalAveragePooling2D()(input_feature)
    avg_pool = Reshape((1, 1, channel))(avg_pool)
    assert avg_pool._keras_shape[1:] == (1, 1, channel)
    avg_pool = Permute((3, 1, 2))(avg_pool)
    avg_pool = Lambda(squeeze)(avg_pool)
    avg_pool = shared_layer_one(avg_pool)
    avg_pool = Lambda(unsqueeze)(avg_pool)
    avg_pool = Permute((2, 3, 1))(avg_pool)
    assert avg_pool._keras_shape[1:] == (1, 1, channel )


    max_pool = GlobalMaxPooling2D()(input_feature)
    max_pool = Reshape((1, 1, channel))(max_pool)
    assert max_pool._keras_shape[1:] == (1, 1, channel)
    max_pool = Permute((3, 1, 2))(max_pool)
    max_pool = Lambda(squeeze)(max_pool)
    max_pool = shared_layer_one(max_pool)
    max_pool = Lambda(unsqueeze)(max_pool)
    max_pool = Permute((2, 3, 1))(max_pool)
    assert max_pool._keras_shape[1:] == (1, 1, channel)


    eca_feature = Add()([avg_pool, max_pool])
    eca_feature = Activation('sigmoid')(eca_feature)

    if K.image_data_format() == "channels_first":
        eca_feature = Permute((3, 1, 2))(eca_feature)

    return multiply([input_feature, eca_feature])
def unsqueeze(input):
    return K.expand_dims(input,axis=-1)

def squeeze(input):
    return K.squeeze(input,axis=-1)


