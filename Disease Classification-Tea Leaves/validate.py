from unittest import result
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input
import numpy as np

model = load_model('disease_detection.h5')

img = image.load_img(r"C:/Users/Hareesh/Desktop/DataSets/TeaLeaves_Diseases/tea sickness dataset/healthy/UNADJUSTEDNONRAW_thumb_239.jpg", target_size=(224,224))
x = image.img_to_array(img)
x = np.expand_dims(x,axis=0)
img_data = preprocess_input(x)
output = np.argmax(model.predict(img_data), axis=1)
index = ['Anthracnose','algal leaf','bird eye spot','brown blight','gray light','healthy','red leaf spot','white spot']
result = str(index[output[0]])
print(result)
