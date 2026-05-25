import tensorflow as tf

model = tf.keras.models.load_model("dynamic_sign_model.keras")

model.save("dynamic_sign_model.h5")

print("Converted!")