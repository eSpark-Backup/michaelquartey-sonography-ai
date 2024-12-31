from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image

model = load_model('model.keras')

def preprocess_image(image_path, target_size=(256, 256)):
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

test_image_path = 'Dataset_BUSI_with_GT/normal/normal (1).png'
preprocessed_image = preprocess_image(test_image_path)

predictions = model.predict(preprocessed_image)

predicted_class_index = np.argmax(predictions, axis=1)[0]

# class_indices = train_generator.class_indices
class_labels = {
    0: 'Benign',
    1: 'Malignant',
    2: 'Normal',
}
predicted_class_label = class_labels[predicted_class_index]

print(f"Predicted Class: {predicted_class_label}")
print(f"Prediction Confidence: {predictions[0][predicted_class_index]:.2f}")
