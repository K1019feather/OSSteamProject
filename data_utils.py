from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (128, 128)
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2

def get_data_generator(rescale=1./255, augment=True):
    if augment:
        return ImageDataGenerator(
            rescale=rescale,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            validation_split=VALIDATION_SPLIT
        )
    else:
        return ImageDataGenerator(
            rescale=rescale,
            validation_split=VALIDATION_SPLIT
        )

def get_data_loaders(data_dir, img_size=IMG_SIZE, batch_size=BATCH_SIZE):
    train_gen = get_data_generator(augment=True)
    val_gen = get_data_generator(augment=False)

    train_data = train_gen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )

    val_data = val_gen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )

    return train_data, val_data
