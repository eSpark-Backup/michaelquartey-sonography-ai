import os, itertools
import numpy as np, pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras import regularizers
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import resample


data_dir = 'C:/Users/mahas/Documents/GitHub/michaelquartey-sonography-ai/Breast_Tumor_Classification/dataset'
filepaths = []
labels = []

folds = os.listdir(data_dir)
for fold in folds:
    foldpath = os.path.join(data_dir, fold)
    filelist = os.listdir(foldpath)
    for file in filelist:
        fpath = os.path.join(foldpath, file)
        filepaths.append(fpath)
        labels.append(fold)

Fseries = pd.Series(filepaths, name= 'filepaths')
Lseries = pd.Series(labels, name='labels')
df = pd.concat([Fseries, Lseries], axis= 1)

strat = df['labels']
train_df, test_df = train_test_split(df, train_size= 0.8, shuffle=True, random_state=123, stratify=strat)

batch_size = 8
img_size = (224, 224)
channels = 3
img_shape = (img_size[0], img_size[1], channels)

benign = train_df[train_df['labels'] == 'benign']
malignant = train_df[train_df['labels'] == 'malignant']
normal = train_df[train_df['labels'] == 'normal']

malignant_upsampled = resample(malignant, replace=True, n_samples=400, random_state=123)
normal_upsampled = resample(normal, replace=True, n_samples=400, random_state=123)

balanced_train_df = pd.concat([benign, malignant_upsampled, normal_upsampled])

tr_gen = ImageDataGenerator(rotation_range=20, zoom_range=0.15, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
                                horizontal_flip=True, fill_mode="nearest")
ts_gen = ImageDataGenerator()

train_gen = ts_gen.flow_from_dataframe(balanced_train_df, target_size=(224, 224), x_col='filepaths', y_col='labels',
                                                class_mode='categorical', shuffle=True, batch_size=batch_size)

# train_gen = tr_gen.flow_from_dataframe(train_df, x_col= 'filepaths', y_col= 'labels', target_size= img_size, 
#                                        class_mode= 'categorical', color_mode= 'rgb', shuffle= True, batch_size= batch_size)

test_gen = ts_gen.flow_from_dataframe(test_df, x_col= 'filepaths', y_col= 'labels', target_size= img_size, 
                                      class_mode= 'categorical', color_mode= 'rgb', shuffle= False, batch_size= batch_size)

base_model = tf.keras.applications.efficientnet.EfficientNetB3(include_top=False, weights='imagenet', 
                                                               input_shape=img_shape, pooling='max')

model = Sequential([
    base_model,
    BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001),
    Dense(256, kernel_regularizer=regularizers.l2(0.016), activity_regularizer=regularizers.l1(0.006),
                bias_regularizer=regularizers.l1(0.006), activation='relu'),
    Dropout(rate=0.45, seed=123),
    Dense(3, activation='softmax')
])

model.compile(Adamax(learning_rate= 0.001), loss= 'categorical_crossentropy', metrics= ['accuracy'])

epochs = 10
history = model.fit(x=train_gen, epochs=epochs, verbose=1, validation_data=test_gen, shuffle=False)

train_score = model.evaluate(train_gen, verbose=1)
test_score = model.evaluate(test_gen, verbose=1)

print("Train Loss: ", train_score[0])
print("Train Accuracy: ", train_score[1])
print("Test Loss: ", test_score[0])
print("Test Accuracy: ", test_score[1])

preds = model.predict(test_gen)
y_pred = np.argmax(preds, axis=1)

g_dict = test_gen.class_indices
classes = list(g_dict.keys())

print(classification_report(test_gen.classes, y_pred, target_names=classes))

model.save('classification_model_3.h5')