import datetime
import tensorflow as tf
import plotly.graph_objects as go
from tensorflow import keras
from keras import layers
import numpy as np
class_names = ['apple_pie',
                'baby_back_ribs',
                'baklava',
                'beef_carpaccio',
                'beef_tartare',
                'beet_salad',
                'beignets',
                'bibimbap',
                'bread_pudding',
                'breakfast_burrito',
                'bruschetta',
                'caesar_salad',
                'cannoli',
                'caprese_salad',
                'carrot_cake',
                'ceviche',
                'cheesecake',
                'cheese_plate',
                'chicken_curry',
                'chicken_quesadilla',
                'chicken_wings',
                'chocolate_cake',
                'chocolate_mousse',
                'churros',
                'clam_chowder',
                'club_sandwich',
                'crab_cakes',
                'creme_brulee',
                'croque_madame',
                'cup_cakes',
                'deviled_eggs',
                'donuts',
                'dumplings',
                'edamame',
                'eggs_benedict',
                'escargots',
                'falafel',
                'filet_mignon',
                'fish_and_chips',
                'foie_gras',
                'french_fries',
                'french_onion_soup',
                'french_toast',
                'fried_calamari',
                'fried_rice',
                'frozen_yogurt',
                'garlic_bread',
                'gnocchi',
                'greek_salad',
                'grilled_cheese_sandwich',
                'grilled_salmon',
                'guacamole',
                'gyoza',
                'hamburger',
                'hot_and_sour_soup',
                'hot_dog',
                'huevos_rancheros',
                'hummus',
                'ice_cream',
                'lasagna',
                'lobster_bisque',
                'lobster_roll_sandwich',
                'macaroni_and_cheese',
                'macarons',
                'miso_soup',
                'mussels',
                'nachos',
                'omelette',
                'onion_rings',
                'oysters',
                'pad_thai',
                'paella',
                'pancakes',
                'panna_cotta',
                'peking_duck',
                'pho',
                'pizza',
                'pork_chop',
                'poutine',
                'prime_rib',
                'pulled_pork_sandwich',
                'ramen',
                'ravioli',
                'red_velvet_cake',
                'risotto',
                'samosa',
                'sashimi',
                'scallops',
                'seaweed_salad',
                'shrimp_and_grits',
                'spaghetti_bolognese',
                'spaghetti_carbonara',
                'spring_rolls',
                'steak',
                'strawberry_shortcake',
                'sushi',
                'tacos',
                'takoyaki',
                'tiramisu',
                'tuna_tartare',
                'waffles']


def get_classes():
    return class_names

def load_and_prep(image, shape=224, scale=False):
    image = tf.image.decode_image(image, channels=3)
    image = tf.image.resize(image, size=([shape, shape]))
    if scale:
        image = image/255.
    return image



def returnModel(select=1):
    if select ==1:
        model_1 = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(filters=10, 
                                    kernel_size=3, # can also be (3, 3)
                                    activation="relu", 
                                    input_shape=(224, 224, 3)), # first layer specifies input shape (height, width, colour channels)
            tf.keras.layers.Conv2D(10, 3, activation="relu"),
            tf.keras.layers.MaxPool2D(pool_size=2, # pool_size can also be (2, 2)
                                        padding="valid"), # padding can also be 'same'
            tf.keras.layers.Conv2D(10, 3, activation="relu"),
            tf.keras.layers.Conv2D(10, 3, activation="relu"), # activation='relu' == tf.keras.layers.Activations(tf.nn.relu)
            tf.keras.layers.MaxPool2D(2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(100, activation='relu'), # increase number of neurons from 4 to 100 (for each layer)
            tf.keras.layers.Dense(100, activation='relu'),
            tf.keras.layers.Dense(100, activation='relu'), # add an extra layer
            tf.keras.layers.Dense(1, activation="sigmoid") # binary activation output
        ])
    if select ==2:
        # Create base model
        input_shape = (224, 224, 3)
        base_model = tf.keras.applications.EfficientNetB0(include_top=False )
        base_model.trainable = False
        # Input and Data Augmentation
        inputs = layers.Input(shape=input_shape, name="input_layer")
        x = base_model(inputs)

        x = layers.GlobalAveragePooling2D(name="pooling_layer")(x)
        x = layers.Dropout(.3)(x)

        x = layers.Dense(10)(x)
        outputs = layers.Activation("softmax")(x)
        model_1 = tf.keras.Model(inputs, outputs)

    if select ==3:
        # Create base model
        input_shape = (224, 224, 3)
        base_model = tf.keras.applications.EfficientNetB1(include_top=False )
        base_model.trainable = True
        # Input and Data Augmentation
        inputs = layers.Input(shape=input_shape, name="input_layer")
        x = base_model(inputs)

        x = layers.GlobalAveragePooling2D(name="pooling_layer")(x)
        x = layers.Dropout(.3)(x)

        x = layers.Dense(101)(x)
        outputs = layers.Activation("softmax")(x)
        model_1 = tf.keras.Model(inputs, outputs)



    return model_1