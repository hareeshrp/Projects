from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator

path = "C:/Users/Hareesh/Desktop/DataSets/TeaLeaves_Diseases/tea sickness dataset"
imageSize = [224, 224]

vgg = VGG16(input_shape=imageSize + [3], weights='imagenet', include_top=False)

for layer in vgg.layers:
    layer.trainable = False

FlattenedLayer = Flatten()(vgg.output)
OutputLayer = Dense(8, activation='softmax')(FlattenedLayer)

model = Model(inputs=vgg.input, outputs=OutputLayer)
print(model.summary())

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'], run_eagerly=True)

image_generator = ImageDataGenerator(validation_split=0.2, rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
train_data_gen = image_generator.flow_from_directory(directory=path, target_size=(224, 224), batch_size=32, class_mode='categorical', interpolation='bicubic', subset='training')
val_data_gen = image_generator.flow_from_directory(directory=path, target_size=(224, 224), batch_size=32, class_mode='categorical', interpolation='bicubic', subset='validation')

print(train_data_gen)
print(val_data_gen)
stats = model.fit(train_data_gen, validation_data=val_data_gen, epochs=20, validation_steps=len(val_data_gen)//32)

model.save('disease_detection1.h5')
