import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

data = np.load("../sign_landmarks_data.npy", allow_pickle=True).item()

X = np.array(data["data"])
y = np.array(data["labels"])

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

y_categorical = to_categorical(y_encoded)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(42,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(set(y)), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(X, y_categorical, epochs=15, batch_size=32)

model.save("../gesture_web/sign_model/sign_model.keras")
np.save("../gesture_web/sign_model/sign_labels.npy", encoder.classes_)

print("Sign model trained and saved!")