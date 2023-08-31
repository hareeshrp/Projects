import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping
import os
import numpy as np
import cv2
from glob import glob
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from patchify import patchify
from vit import ViT

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#Hyperparameters
hp = {}
hp['image_size'] = 200
hp['num_channels'] = 3
hp['patch_size'] = 25
hp['num_patches'] = (hp['image_size']**2) // (hp['patch_size']**2)
hp['flat_patches_shape'] = (hp['num_patches'], hp['patch_size']*hp['patch_size']*hp['num_channels'])

hp['batch_size'] = 32
hp['lr'] = 1e-4
hp['num_epochs'] = 500
hp['num_classes'] = 4
hp['class_names'] = ['Covid','Lung Opacity', 'Normal', 'Viral Pneumonia']

hp["num_layers"] = 12
hp["hidden_dims"] = 768
hp["mlp_dim"] = 3072
hp["num_heads"] = 12
hp["dropout_rate"] = 0.1

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_data(path, split=0.1):
    images = shuffle(glob(os.path.join(path, "*", "*.png")))
    split_size = int(len(images) * split)
    X_train, X_valid = train_test_split(images, test_size=split_size, random_state=42)
    X_train, X_test = train_test_split(X_train, test_size=split_size, random_state=42)

    return X_train, X_valid, X_test

def process_image_label(path):
    path = path.decode()
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (hp['image_size'], hp['image_size']))
    image = image/255.0
    #print(image.shape)

    patch_shape = (hp['patch_size'], hp['patch_size'], hp['num_channels'])
    patches = patchify(image, patch_shape, hp['patch_size'])
    patches =  np.reshape(patches, hp['flat_patches_shape'])
    patches = patches.astype(np.float32)


    class_name = path.split("\\")[-2]
    class_idx = hp['class_names'].index(class_name)
    class_idx = np.array(class_idx, dtype=np.int32)
    print(class_idx)
    return patches, class_idx


def parse(path):
    patches, labels = tf.numpy_function(process_image_label, [path], [tf.float32, tf.int32])
    labels = tf.one_hot(labels, hp['num_classes'])

    patches.set_shape(hp['flat_patches_shape'])
    labels.set_shape(hp['num_classes'])

    return patches, labels


def tf_dataset(images, batch=32):
    ds = tf.data.Dataset.from_tensor_slices((images))
    ds = ds.map(parse).batch(batch).prefetch(8)
    return ds



if __name__ == "__main__":
    np.random.seed(42)
    tf.random.set_seed(42)

    create_dir("files")

    dataset_path = "C:/Users/Hareesh/Desktop/DataSets/Covid19-RadioGraphy/COVID-19_Radiography_Dataset/Data"
    model_path = os.path.join("files", 'model.h5')
    csv_path = os.path.join('files', 'log.csv')

    X_train, X_valid, X_test = load_data(dataset_path)

    #process_image_label(X_train[0])
    train_ds = tf_dataset(X_train, batch=hp['batch_size'])
    valid_ds = tf_dataset(X_valid, batch=hp['batch_size'])

    #Model building

    model = ViT(hp)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(hp['lr'], clipvalue=1.0),
        metrics=['accuracy']
    )
    
    callbacks = [
        ModelCheckpoint(model_path, monitor='val_loss', verbose=1, save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=1e-10, verbose=1),
        CSVLogger(csv_path),
        EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=False),
    ]

    model.fit(
        train_ds,
        epochs=hp['num_epochs'],
        validation_data=valid_ds,
        callbacks=callbacks
    )
    

