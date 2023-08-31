import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Embedding, Layer, Concatenate, LayerNormalization, MultiHeadAttention, Add, Dense, Dropout
from tensorflow.keras.models import Model
#from tensorflow.keras.layers import Model

@tf.keras.utils.register_keras_serializable()
class ClassToken(Layer):
    def __init__(self):
        super().__init__()

    def build(self, input_shape):
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(
            initial_value=w_init(shape=(1, 1, input_shape[-1]),dtype=tf.float32 ),
            trainable=True
        )

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        hidden_dim = self.w.shape[-1]
        cls = tf.broadcast_to(self.w, [batch_size, 1, hidden_dim])
        cls = tf.cast(cls, dtype=inputs.dtype)
        return cls
    

def mlp(x, cf):
    x = Dense(cf['mlp_dim'], activation='gelu')(x)
    x = Dropout(cf['dropout_rate'])(x)
    x = Dense(cf['hidden_dims'])(x)
    x = Dropout(cf['dropout_rate'])(x)

    return x


def transformer_encoder(x, cf):
    skip_1 = x
    x = LayerNormalization()(x)
    x = MultiHeadAttention(
        num_heads = cf['num_heads'],
        key_dim = cf['hidden_dims'],
    )(x, x)
    x = Add()([x, skip_1])
    skip_2 = x
    x = LayerNormalization()(x)
    x = mlp(x, cf)
    x = Add()([x, skip_2])

    return x


def ViT(cf):
    input_shape = (cf['num_patches'], cf['patch_size']**2 * cf['num_channels'])
    inputs = Input(input_shape)
    
    #Patch embedding
    patch_embed = Dense(cf['hidden_dims'])(inputs)
    
    #Position embedding
    positions = tf.range(start=0, limit=cf['num_patches'], delta=1)
    pos_embed = Embedding(input_dim=cf['num_patches'], output_dim=cf['hidden_dims'])(positions)
    
    embed = patch_embed + pos_embed
    
    #Adding class token

    token = ClassToken()(embed)
    x = Concatenate(axis=1)([token, embed])

    for _ in range(cf['num_layers']):
        x = transformer_encoder(x, cf)

    x = LayerNormalization()(x)
    x = x[:, 0, :]

    x = Dense(cf['num_classes'], activation='softmax')(x)

    model = Model(inputs, x)
    
    return model


    

if __name__ == "__main__":
    config = {}
    config['num_layers'] = 12
    config['hidden_dims'] = 768
    config['mlp_dim'] = 3072
    config['num_heads'] = 12
    config['dropout_rate'] = 0.1
    config['num_patches'] = 256
    config['patch_size'] = 32
    config['num_channels'] = 3
    config['num_classes'] = 4
    
    model = ViT(config)
    model.summary()