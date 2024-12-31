import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# TensorFlow
import tensorflow as tf

# Keras
from keras.layers import Flatten, Dense
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.applications import DenseNet121
from keras.applications.densenet import DenseNet121

# scikit-learn
from sklearn.model_selection import train_test_split

train_data = 'Dataset_BUSI_with_GT'

pd.DataFrame(
    os.listdir(train_data),
    columns=['File Name']
)

train_files = [i for i in glob.glob(train_data + "/*/*")]
np.random.shuffle(train_files)
labels = [os.path.dirname(i).split("/")[-1] for i in train_files]
data = zip(train_files, labels)
training_data = pd.DataFrame(data, columns=["Path", "Label"])
print(training_data)

data_paths = []
labels = []

main_dirs = os.listdir(train_data)
for folder_name in main_dirs:
    folder_path = os.path.join(train_data, folder_name)
    files = os.listdir(folder_path)
    for file in files:
        file_path = os.path.join(folder_path, file)
        data_paths.append(file_path)
        labels.append(folder_name)

batch_size = 16
image_size = (256, 256)
num_channels = 3
image_shape = (image_size[0], image_size[1], num_channels)

data_dirs = pd.Series(data_paths, name='Path')

classes = pd.Series(labels, name='Label')

def preprocess_image(img):
    return img

train_df, val_test_df = train_test_split(
    training_data,
    train_size=0.8,
    shuffle=True,
    random_state=123
)

val_df, test_df = train_test_split(
    val_test_df,
    train_size=0.5,
    shuffle=True,
    random_state=123
)

datagen = ImageDataGenerator(
    preprocessing_function=preprocess_image,
    rescale=1.0/255
)

train_df['Label'] = train_df['Label'].astype(str)
val_df['Label'] = val_df['Label'].astype(str)

train_generator = datagen.flow_from_dataframe(
    train_df,
    x_col='Path',
    y_col='Label',
    target_size=image_size,
    class_mode='categorical',
    color_mode='rgb',
    shuffle=True,
    batch_size=batch_size
)

valid_generator = datagen.flow_from_dataframe(
    val_df,
    x_col='Path',
    y_col='Label',
    target_size=image_size,
    class_mode='categorical',
    color_mode='rgb',
    shuffle=True,
    batch_size=batch_size
)

base_model = DenseNet121(
    weights='densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5',
    include_top=False,
    input_shape=(256, 256, 3)
)
for layer in base_model.layers:
    layer.trainable = False

model = Sequential()
model.add(base_model)
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dense(1024, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.summary()

checkpoint_path = "/working/model.keras"

checkpoint = ModelCheckpoint(
    checkpoint_path,
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=20,
    validation_data=valid_generator,
    validation_steps=valid_generator.samples // valid_generator.batch_size,
    callbacks=[checkpoint]
)
train_history = pd.DataFrame(history.history)

train_history['Epoch'] = range(1, len(train_history) + 1)
train_history = train_history[['Epoch', 'loss', 'accuracy', 'val_loss', 'val_accuracy']]
print(train_history)

#     Epoch   loss	    accuracy	val_loss    val_accuracy
#  0    1	2.244131	0.729535	0.441344	0.833333
#  1    2	0.344283	0.812500	0.345094	0.785714
#  2    3	0.313649	0.874799	0.361469	0.854167
#  3    4	0.222720	0.875000	0.233024	0.857143
#  4    5	0.215100	0.923756	0.496515	0.840278
#  5    6	0.063823	0.937500	0.745814	0.857143
#  6    7	0.138316	0.950241	0.465127	0.861111
#  7    8	0.034011	1.000000	0.564421	0.857143
#  8    9	0.090554	0.972713	0.706265	0.861111
#  9    10	0.045931	1.000000	0.752238	0.857143
# 10	11	0.105637	0.963884	0.505534	0.854167
# 11	12	0.005680	1.000000	0.539086	0.785714
# 12	13	0.038378	0.990369	0.637371	0.902778
# 13	14	0.102371	0.937500	1.422136	0.857143
# 14	15	0.059358	0.980738	0.381454	0.902778
# 15	16	0.020186	1.000000	1.421154	0.785714
# 16	17	0.082295	0.975923	0.690594	0.868056
# 17	18	0.000155	1.000000	0.227391	0.928571
# 18	19	0.102726	0.972713	0.557599	0.868056
# 19	20	0.157104	0.937500	0.131427	0.928571

train_loss, train_accuracy = model.evaluate(train_generator, steps=train_generator.samples // train_generator.batch_size)
val_loss, val_accuracy = model.evaluate(valid_generator, steps=valid_generator.samples // valid_generator.batch_size)
train_accuracy_percentage = train_accuracy * 100
val_accuracy_percentage = val_accuracy * 100
evaluation_results = pd.DataFrame({
    'Set': ['Train', 'Validation'],
    'Loss': [train_loss, val_loss],
    'Accuracy': [f'{train_accuracy_percentage:.2f}%', f'{val_accuracy_percentage:.2f}%']
})
print(evaluation_results)

# 	Set	        Loss	    Accuracy
# 0	Train	    0.012806	99.68%
# 1	Validation	0.383065	92.36%