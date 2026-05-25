import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

print("Starting training...")

data = np.load("landmarks_data.npy", allow_pickle=True).item()

X = np.array(data["data"])
y = np.array(data["labels"])

print("Total samples:", len(y))

encoder = LabelEncoder()
y_numbers = encoder.fit_transform(y)

y_ready = to_categorical(y_numbers)

X_train, X_test, y_train, y_test = train_test_split(
X,
y_ready,
test_size=0.2
)

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Dense(
64,
activation="relu",
input_shape=(X.shape[1],)
))

model.add(tf.keras.layers.Dense(
y_ready.shape[1],
activation="softmax"
))


model.compile(
optimizer="adam",
loss="categorical_crossentropy",
metrics=["accuracy"]
)


model.fit(
X_train,
y_train,
epochs=30
)

model.save("hand_gesture_model.keras")
np.save("label_encoder.npy", encoder.classes_)

print("Done! Model saved.")