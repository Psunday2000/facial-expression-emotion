import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    BatchNormalization,
    Flatten,
    LeakyReLU,
    Dense,
    Dropout,
)
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau

import os
train_path = (
    "C:/Users/DELL/OneDrive/Documents/GitHub/facial-expression-emotion/fer2013/train"
)
test_path = (
    "C:/Users/DELL/OneDrive/Documents/GitHub/facial-expression-emotion/fer2013/test"
)
x = plt.imread(
    "C:/Users/DELL/OneDrive/Documents/GitHub/facial-expression-emotion/fer2013/test/happy/PrivateTest_6908247.jpg"
)
x.shape
datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1 / 255.0)
train = datagen.flow_from_directory(
    train_path,
    target_size=(48, 48),
    class_mode="sparse",
    seed=1,
    color_mode="grayscale",
    batch_size=128,
)
test = datagen.flow_from_directory(
    test_path,
    target_size=(48, 48),
    class_mode="sparse",
    seed=1,
    color_mode="grayscale",
    batch_size=128,
)
dir(train)
print(train.class_indices)
emotion_dict = {
    0: "Angry",
    1: "Disgusted",
    2: "Fearful",
    3: "Happy",
    4: "Neutral",
    5: "Sad",
    6: "Surprised",
}
# Get the next batch from the train iterator
x, y = next(train)

# Print the shapes of x and y
print(x.shape, y.shape)

# Reset the iterator to the beginning of the dataset (optional)
train.reset()
model = Sequential(
    [
        Conv2D(32, (3, 3), input_shape=(48, 48, 1), padding="same"),
        LeakyReLU(),
        Conv2D(32, (3, 3), padding="same"),
        LeakyReLU(),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.2),
        Conv2D(64, (3, 3), padding="same"),
        LeakyReLU(),
        Conv2D(64, (3, 3), padding="same"),
        LeakyReLU(),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.3),
        Conv2D(128, (3, 3), padding="same"),
        LeakyReLU(),
        Conv2D(128, (3, 3), padding="same"),
        LeakyReLU(),
        Conv2D(128, (3, 3), padding="same"),
        LeakyReLU(),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.4),
        Flatten(),
        #     tf.keras.layers.GlobalAveragePooling2D(),
        #     Dropout(0.4),
        Dense(128, activation="relu"),
        Dropout(0.4),
        #     Dense(64, activation="relu"),
        Dense(len(train.class_indices), activation="softmax"),
    ]
)
model.summary()
model.compile(
    optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
net = Sequential(name="DCNN")

net.add(
    Conv2D(
        filters=64,
        kernel_size=(5, 5),
        input_shape=(48, 48, 1),
        activation="elu",
        padding="same",
        kernel_initializer="he_normal",
        name="conv2d_1",
    )
)
net.add(BatchNormalization(name="batchnorm_1"))
net.add(
    Conv2D(
        filters=64,
        kernel_size=(5, 5),
        activation="elu",
        padding="same",
        kernel_initializer="he_normal",
        name="conv2d_2",
    )
)
net.add(BatchNormalization(name="batchnorm_2"))

net.add(MaxPooling2D(pool_size=(2, 2), name="maxpool2d_1"))
net.add(Dropout(0.4, name="dropout_1"))

net.add(
    Conv2D(
        filters=128,
        kernel_size=(3, 3),
        activation="elu",
        padding="same",
        kernel_initializer="he_normal",
        name="conv2d_3",
    )
)
net.add(BatchNormalization(name="batchnorm_3"))
net.add(
    Conv2D(
        filters=128,
        kernel_size=(3, 3),
        activation="elu",
        padding="same",
        kernel_initializer="he_normal",
        name="conv2d_4",
    )
)
net.add(BatchNormalization(name="batchnorm_4"))

net.add(MaxPooling2D(pool_size=(2, 2), name="maxpool2d_2"))
net.add(Dropout(0.4, name="dropout_2"))

net.add(
    Conv2D(
        filters=256,
        kernel_size=(3, 3),
        activation="elu",
        padding="same",
        kernel_initializer="he_normal",
        name="conv2d_5",
    )
)
net.add(BatchNormalization(name="batchnorm_5"))
net.add(
    Conv2D(
        filters=256,
        kernel_size=(3, 3),
        activation="elu",
        padding="same",
        kernel_initializer="he_normal",
        name="conv2d_6",
    )
)
net.add(BatchNormalization(name="batchnorm_6"))

net.add(MaxPooling2D(pool_size=(2, 2), name="maxpool2d_3"))
net.add(Dropout(0.5, name="dropout_3"))

net.add(Flatten(name="flatten"))

net.add(Dense(128, activation="elu", kernel_initializer="he_normal", name="dense_1"))
net.add(BatchNormalization(name="batchnorm_7"))

net.add(Dropout(0.6, name="dropout_4"))

net.add(Dense(7, activation="softmax", name="out_layer"))

net.compile(
    loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
)

net.summary()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
early_stopping = EarlyStopping(
    monitor="val_accuracy",
    min_delta=0.00005,
    patience=11,
    verbose=1,
    restore_best_weights=True,
)

lr_scheduler = ReduceLROnPlateau(
    monitor="val_accuracy",
    factor=0.5,
    patience=7,
    min_lr=1e-7,
    verbose=1,
)

callbacks = [
    early_stopping,
    lr_scheduler,
]
model.fit(train, validation_data=test, epochs=20, callbacks=callbacks)
model.save("facialmodel.h5")
x, y = next(test)
preds = model.predict(x)
idx = np.argmax(preds, axis=1)
# idx.shape
print("correct prediction:", np.sum((y == idx) * 1) / 128)
fig = plt.figure(1, (14, 14))

k = 0
for j in range(49):
    px = x[j]
    k += 1
    ax = plt.subplot(7, 7, k)
    ax.imshow(px, cmap="gray")
    ax.set_xticks([])
    ax.set_yticks([])

    if emotion_dict[y[j]] == emotion_dict[idx[j]]:
        for axis in ["top", "bottom", "left", "right"]:
            ax.spines[axis].set_linewidth(5)
            ax.spines[axis].set_color("green")
        ax.set_title(emotion_dict[idx[j]])

    else:
        for axis in ["top", "bottom", "left", "right"]:
            ax.spines[axis].set_linewidth(5)
            ax.spines[axis].set_color("red")
        ax.set_title("P:" + emotion_dict[idx[j]] + " C:" + emotion_dict[y[j]])
    plt.tight_layout()
test.reset()
