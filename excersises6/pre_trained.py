from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np

model = VGG16(weights='imagenet', classes=1000)
img_path = 'sampleImages/banana.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, 0)
x = preprocess_input(x)
preds = model.predict(x)
decoded_preds = decode_predictions(preds, top=5)[0]
print('Predicted: ')
for p in decoded_preds:
    print(p[1], " : ", p[2])
