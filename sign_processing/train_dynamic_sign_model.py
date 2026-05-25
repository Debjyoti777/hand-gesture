import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

data = np.load("../dynamic_sign_data.npy", allow_pickle=True).item()

X = np.array(data["data"])
y = np.array(data["labels"])

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
y_cat = to_categorical(y_encoded)

model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(128, input_shape=(20, 84)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(set(y)), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(X, y_cat, epochs=20, batch_size=16)

model.save("../gesture_web/sign_model/dynamic_sign_model.keras")
np.save("../gesture_web/sign_model/dynamic_sign_labels.npy",
        encoder.classes_)

print("Dynamic model trained!")