# import all packages
from tensorflow.keras.models import Model
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
import numpy as np
import os
from imutils import paths
import matplotlib.pyplot as plt
import tensorflow as tf
tf.config.list_physical_devices('GPU')

EPOCHS = 20
INITIAL_LEARNING_RATE = 1e-4
BATCH_SIZE = 32

print(">>>> Loading Dataset <<<<")

DIRECTORY = r".\Dataset"
CATEGORIES = ["with_mask", "without_mask"]
data = []
labels = []

for category in CATEGORIES:
    path = os.path.join(DIRECTORY, category)
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        image = load_img(img_path, target_size=(224, 224))
        image = img_to_array(image)
        image = preprocess_input(image)

        data.append(image)
        labels.append(category)

# convert labels to Numeric Catagorical
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

data = np.array(data, dtype="float32")
labels = np.array(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels,
                                                  test_size=0.20, stratify=labels, random_state=42)

# Image generator using data augmentation
aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest")


# CNN: MobileNetV2
baseModel = MobileNetV2(weights="imagenet", include_top=False,
                        input_tensor=Input(shape=(224, 224, 3)))

# Create models
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)
model = Model(inputs=baseModel.input, outputs=headModel)

for layer in baseModel.layers:
    layer.trainable = False
with tf.device('/GPU:0'):
    # compiling the model
    print("MODEL Compilation is started >>")
    opt = Adam(lr=INITIAL_LEARNING_RATE, decay=INITIAL_LEARNING_RATE / EPOCHS)
    model.compile(loss="binary_crossentropy", optimizer=opt,
                  metrics=["accuracy"])

    # training the model network
    print("Training Head Started >>")
    H = model.fit(
        aug.flow(trainX, trainY, batch_size=BATCH_SIZE),
        steps_per_epoch=len(trainX) // BATCH_SIZE,
        validation_data=(testX, testY),
        validation_steps=len(testX) // BATCH_SIZE,
        epochs=EPOCHS)

    # predictions
    print("Testing Starting >>")
    predIdxs = model.predict(testX, batch_size=BATCH_SIZE)
    predIdxs = np.argmax(predIdxs, axis=1)
    print(classification_report(testY.argmax(axis=1), predIdxs,
                                target_names=lb.classes_))

# saving the model
print("Model Saving to the loacl directory")
model.save("Mask_Detector_Model.model", save_format="h5")

# ploting the accuracy and loss
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Modle Accuracy and Loss")
plt.xlabel("Epoch")
plt.ylabel("Accuracy & Loss")
plt.legend(loc="lower right")
plt.savefig("Accuracy&Loss.jpg")
