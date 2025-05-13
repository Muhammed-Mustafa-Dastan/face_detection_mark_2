import tensorflow as tf
from tensorflow.keras import layers, Model

class L1Dist(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)

def get_siamese_model(input_shape=(160, 160, 3)):
    input_a = layers.Input(input_shape)
    input_b = layers.Input(input_shape)
    base_model = get_embedding_model(input_shape)
    emb_a = base_model(input_a)
    emb_b = base_model(input_b)
    l1_layer = L1Dist()(emb_a, emb_b)
    output = layers.Dense(1, activation='sigmoid')(l1_layer)
    model = Model(inputs=[input_a, input_b], outputs=output)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def get_embedding_model(input_shape=(160, 160, 3)):
    # Basit bir CNN, gerçek projede daha iyi bir model kullanılabilir
    inputs = layers.Input(input_shape)
    x = layers.Conv2D(32, 3, activation='relu')(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, activation='relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(x)
    return Model(inputs, outputs)
