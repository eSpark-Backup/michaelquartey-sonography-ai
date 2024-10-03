from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

model = load_model('classification_model_3.h5')

img_path = 'C:/Users/mahas/Documents/GitHub/michaelquartey-sonography-ai/Breast_Tumor_Classification/dataset/benign/benign (106).png'
img = image.load_img(img_path, target_size=(224, 224))  # Resizing the image
img_array = image.img_to_array(img)  # Converting image to array
img_array = np.expand_dims(img_array, axis=0)  # Adding batch dimension
img_array /= 255.0  # Normalizing pixel values

predictions = model.predict(img_array, batch_size=1)
predicted_class = np.argmax(predictions, axis=1)

class_labels = {0:'benign', 1:'malignant', 2:'normal'}

print(predictions)
predicted_label = class_labels[predicted_class[0]]
print(f"The predicted class is: {predicted_label}")