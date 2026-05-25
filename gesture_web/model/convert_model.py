import tensorflow as tf

model = tf.keras.models.load_model("hand_gesture_model.keras")

model.save("hand_gesture_model.h5")

print("Converted!")