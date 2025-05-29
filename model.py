from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import TimeDistributed, GRU, Dense, GlobalAveragePooling2D, Dropout

def build_model():
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
    for layer in base_model.layers:
        layer.trainable = False

    model = Sequential([
        TimeDistributed(base_model),
        TimeDistributed(GlobalAveragePooling2D()),
        GRU(64, return_sequences=False),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
