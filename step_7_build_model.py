import tensorflow as tf

# 1. Create the CNN model
model = tf.keras.Sequential([

    # Input image
    tf.keras.layers.Input(shape=(224, 224, 1)),

    # First convolution layer
    tf.keras.layers.Conv2D(
        32,
        (3, 3),
        activation="relu"
    ),

    # Reduce the image size
    tf.keras.layers.MaxPooling2D(
        (2, 2)
    ),

    # Second convolution layer
    tf.keras.layers.Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),

    # Reduce the image size again
    tf.keras.layers.MaxPooling2D(
        (2, 2)
    ),

    # Third convolution layer
    tf.keras.layers.Conv2D(
        128,
        (3, 3),
        activation="relu"
    ),

    # Reduce the image size again
    tf.keras.layers.MaxPooling2D(
        (2, 2)
    ),

    # Turn the feature maps into one long vector
    tf.keras.layers.Flatten(),

    # Fully connected layer
    tf.keras.layers.Dense(
        128,
        activation="relu"
    ),

    # Output layer
    tf.keras.layers.Dense(
        1,
        activation="sigmoid"
    )
])

# 2. Compile the model

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# 3. Display the model
model.summary()