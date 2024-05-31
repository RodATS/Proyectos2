import warnings
import logging
import os
warnings.filterwarnings('ignore')
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import cv2

from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error, classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from vit_keras import vit


IMG_SIZE = 224, 224
BATCH_SIZE = 32
SEED = 999
AUTO = tf.data.AUTOTUNE
tf.random.set_seed(SEED)


# Creating main dataframe contains image paths and their classes.

def create_images_list(path):
    full_path = []
    images = sorted(os.listdir(path))
    for i in images:
        full_path.append(os.path.join(path, i))
    return full_path


train_data_0 = create_images_list('D:/BD_GlaucomaTorch/train_converted_images/0')
train_data_1 = create_images_list('D:/BD_GlaucomaTorch/train_converted_images/1')

val_data_0 = create_images_list('D:/BD_GlaucomaTorch/val_converted_images/0')
val_data_1 = create_images_list('D:/BD_GlaucomaTorch/val_converted_images/1')

test_data_0 = create_images_list('D:/BD_GlaucomaTorch/test_converted_images/0')
test_data_1 = create_images_list('D:/BD_GlaucomaTorch/test_converted_images/1')


# look-up table
classes = {0: 'non-glaucoma', 1: 'glaucoma'}


train_df = pd.concat([pd.DataFrame({ 'image' : train_data_0, 'label': 0 }),
                      pd.DataFrame({ 'image' : train_data_1, 'label': 1 }) ], ignore_index = True)


val_df = pd.concat([pd.DataFrame({ 'image' : val_data_0, 'label': 0 }),
                    pd.DataFrame({ 'image' : val_data_1, 'label': 1 }) ], ignore_index = True)


test_df = pd.concat([pd.DataFrame({ 'image' : test_data_0, 'label': 0 }),
                     pd.DataFrame({ 'image' : test_data_1, 'label': 1 }) ], ignore_index = True)


train_df = shuffle(train_df, random_state = SEED).reset_index(drop = True)
val_df = shuffle(val_df, random_state = SEED).reset_index(drop = True)
test_df = shuffle(test_df, random_state = SEED).reset_index(drop = True)

print("train images -> ", train_df.shape[0])
print("val images -> ", val_df.shape[0])
print("test images -> ", test_df.shape[0])


# Reading -> Resizing -> Normalization
def img_preprocessing(image, label):
    img = tf.io.read_file(image)
    img = tf.io.decode_jpeg(img, channels = 3)
    img = tf.image.resize(img, size = (IMG_SIZE))
    img = tf.cast(img, tf.float32) / 255.0

    return img, label


# Data augmentation
def augmentation(image, label):
    img = tf.image.random_flip_left_right(image, seed = SEED)
    img = tf.image.random_brightness(img, 0.1, seed = SEED)
    img = tf.image.random_contrast(img, 0.2, 0.3, seed = SEED)

    return img, label

# Creating dataset loaders and tf.datasets

train_loader = tf.data.Dataset.from_tensor_slices((train_df['image'], train_df['label']))
train_dataset = (train_loader
                 .map(img_preprocessing, num_parallel_calls = AUTO)
                 .map(augmentation, num_parallel_calls = AUTO)
                 .shuffle(BATCH_SIZE*10)
                 .batch(BATCH_SIZE)
                 .prefetch(AUTO))


val_loader = tf.data.Dataset.from_tensor_slices((val_df['image'], val_df['label']))
val_dataset = (val_loader
               .map(img_preprocessing, num_parallel_calls = AUTO)
               .batch(BATCH_SIZE)
               .prefetch(AUTO))

test_loader = tf.data.Dataset.from_tensor_slices((test_df['image'], test_df['label']))
test_dataset = (test_loader
                .map(img_preprocessing, num_parallel_calls = AUTO)
                .batch(BATCH_SIZE)
                .prefetch(AUTO))



#Modelo VIT
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LayerNormalization, Dropout, Add
from tensorflow.keras.models import Model
import numpy as np
class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.projection_dim = embed_dim // num_heads
        assert self.projection_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.query_dense = Dense(embed_dim)
        self.key_dense = Dense(embed_dim)
        self.value_dense = Dense(embed_dim)
        self.combine_heads = Dense(embed_dim)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)
        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention, (batch_size, -1, self.embed_dim))
        output = self.combine_heads(concat_attention)
        return output

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = tf.keras.Sequential(
            [Dense(ff_dim, activation="gelu"), Dense(embed_dim),]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, GlobalAveragePooling1D, BatchNormalization, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from vit_keras import vit

#numa_layers: capas densas, num_heads: autoatencion, d_model:  Define la dimensionalidad del espacio de los embeddings
#mlp_dim: capa amplía la dimensionalidad antes de reducirla de nuevo, permitiendo una mayor capacidad de representación.
# Definición del modelo ViT-B16
def build_custom_vit_b16(input_shape, num_classes, d_model=768, mlp_dim=3072, dropout=0.1, num_heads=12, num_layers=12):
    inputs = Input(shape=input_shape)
    patch_size = 16
    num_patches = (input_shape[0] // patch_size) * (input_shape[1] // patch_size)
    patches = tf.image.extract_patches(
        images=inputs,
        sizes=[1, patch_size, patch_size, 1],
        strides=[1, patch_size, patch_size, 1],
        rates=[1, 1, 1, 1],
        padding="VALID",
    )
    patches = tf.reshape(patches, [-1, num_patches, patch_size * patch_size * input_shape[2]])

    positions = tf.range(start=0, limit=num_patches, delta=1)
    pos_embed = tf.keras.layers.Embedding(input_dim=num_patches, output_dim=d_model)(positions)
    patch_embed = Dense(d_model)(patches)
    embeddings = patch_embed + pos_embed

    for _ in range(num_layers):
        embeddings = TransformerBlock(d_model, num_heads, mlp_dim, dropout)(embeddings)

    x = LayerNormalization(epsilon=1e-6)(embeddings)
    x = GlobalAveragePooling1D()(x)
    x = Dropout(dropout)(x)

    # Agregar capas adicionales
    x = Flatten()(x)
    x = BatchNormalization()(x)

    x = Dense(256, activation='gelu', kernel_regularizer=l2(0.01))(x)
    #x = BatchNormalization()(x)
    #x = Dropout(dropout)(x)

    x = Dense(64, activation='gelu', kernel_regularizer=l2(0.01))(x)
    ##x = BatchNormalization()(x)
    x = Dropout(dropout)(x)

    x = Dense(32, activation='gelu', kernel_regularizer=l2(0.01))(x)
    #x = BatchNormalization()(x)
   # x = Dropout(dropout)(x)

    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model




# Uso del modelo
input_shape = (224, 224, 3)
num_classes = 2
model = build_custom_vit_b16(input_shape=input_shape, num_classes=num_classes, num_heads=2, num_layers=2)

#0.001, 0.0005 y 0.0001.
# Compilación del modelo
model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=0.001, weight_decay=0.001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['acc'])

# Resumen del modelo
model.summary()

# Entrenamiento del modelo
my_callback = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0.001, patience=5)]
# Suponiendo que `train_dataset` y `val_dataset` están definidos y listos para ser usados
history = model.fit(train_dataset, validation_data=val_dataset, epochs=20, callbacks=my_callback)

# Guardar el modelo completo
model.save('E:/Universidad_UCSP/Proyectos2/PruebasCodigoViT/modelo_VIT_2cabeza_2capas.h5')




