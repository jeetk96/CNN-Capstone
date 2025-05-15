import matplotlib.pyplot as plt

# these are the parameters for training, batch size is the number of training samples processed before the model's internal parameters are updated
my_batch_size = 32

# the height of image
image_height = 224

# the width of image
image_width = 224

# This is the file path that has train_faces !!! Change these paths to where you installed the FaceForensics++ database!!!
my_data_path = r'C:\Users\Jeet\Documents\FF++\train_faces'


# loading the dataset

# this is the dataset for training
from tensorflow.keras.preprocessing import image_dataset_from_directory
data_for_train = image_dataset_from_directory(
    my_data_path,
    validation_split=0.2,
    subset="training",
    seed=145,
    image_size=(image_height, image_width),
    batch_size=my_batch_size
)
# this is the dataset for validation
data_for_val = image_dataset_from_directory(
    my_data_path,
    validation_split=0.2,
    subset="validation",
    seed=145,
    image_size=(image_height, image_width),
    batch_size=my_batch_size
)


# converting labels to categorical format
from tensorflow.keras.utils import to_categorical
def converting_label(data):
    data = data.map(lambda x, y: (x, to_categorical(y, num_classes=2)))
    return data


# converting data
data_for_train = converting_label(data_for_train)
data_for_val = converting_label(data_for_val)

# initialize the base model with EfficientNetV2B0
from tensorflow.keras.applications import EfficientNetV2B0
base_model = EfficientNetV2B0(weights='imagenet', include_top=False, input_shape=(image_height, image_width, 3))
base_model.trainable = False

# new model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
new_model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.5),
    Dense(2, activation='softmax')
])

# compile the model
new_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# this is the summary of the new model
new_model.summary()

# train the model
# epoch represents the number of times the entire dataset is passed through the algorithm
epoch = 50

hist = new_model.fit(data_for_train, validation_data=data_for_val, epochs=epoch)

# this is the evaluation of the new model with validation data
loss, accuracy = new_model.evaluate(data_for_val)
print(f'Validation accuracy: {accuracy * 100:.2f}%')


# plot training and validation accuracy
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(hist.history['accuracy'], label='Train Accuracy')
plt.plot(hist.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.xlim(0, epoch - 1)
plt.legend(loc='lower right')

plt.subplot(1, 2, 2)
plt.plot(hist.history['loss'], label='Train Loss')
plt.plot(hist.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.xlim(0, epoch - 1)
plt.legend(loc='upper right')

# save the plots
plt.tight_layout()
plt.savefig('training_history_EfficientNetV2.png')
plt.show()
