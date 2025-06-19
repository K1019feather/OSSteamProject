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

import os
import numpy as np
from tensorflow.keras.preprocessing.image import array_to_img

def extract_misclassified_samples(model, val_data, class_names, save_dir='misclassified', max_images=20):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    total_checked = 0
    misclassified = 0

    for i in range(len(val_data)):
        images, labels = val_data[i]
        predictions = model.predict(images)
        pred_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(labels, axis=1)

        for j in range(len(images)):
            total_checked += 1
            if pred_classes[j] != true_classes[j]:
                misclassified += 1
                if misclassified > max_images:
                    print(f"✅ 오분류 {max_images}개 저장 완료.")
                    return

                img = array_to_img(images[j])
                fname = f"{misclassified}_P-{class_names[pred_classes[j]]}_T-{class_names[true_classes[j]]}.jpg"
                img.save(os.path.join(save_dir, fname))

    print(f"🔍 전체 {total_checked}개 중 오분류 {misclassified}개 추출 완료.")

