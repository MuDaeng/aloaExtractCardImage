import numpy as np
import tensorflow as tf
from keras import layers

def train_card():
    img_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
                                                              validation_split=0.2)

    data_dir = './train'
    img_height, img_width = 54, 54
    batch_size = 32

    train_set = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    valid_set = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    class_names = train_set.class_names
    print(class_names)

    for image_batch, labels_batch in train_set:
        print(image_batch.shape)
        print(labels_batch.shape)
        break

    AUTOTUNE = tf.data.AUTOTUNE

    train_set = train_set.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    valid_set = valid_set.cache().prefetch(buffer_size=AUTOTUNE)

    normalization_layer = tf.keras.layers.Rescaling(1./255)

    normalized_set = train_set.map(lambda x, y: (normalization_layer(x), y))
    image_batch, labels_batch = next(iter(normalized_set))
    first_image = image_batch[0]
    # Notice the pixel values are now in `[0,1]`.
    print(np.min(first_image), np.max(first_image))

    num_classes = len(class_names)

    model = tf.keras.Sequential([
        layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.summary()

    epochs=50
    history = model.fit(
        train_set,
        validation_data=valid_set,
        epochs=epochs
    )

    model.save('card.h5')