import matplotlib.pyplot as plt
import numpy as np
from keras import backend, models, layers, optimizers, utils


# Function to extract data from mnist.npz dataset
def load_data():
    f = np.load('dataset/mnist.npz')
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']
    f.close()
    return (x_train, y_train), (x_test, y_test)


if backend.backend() == 'tensorflow':
    backend.set_image_dim_ordering("th")

# Extracting and splitting dataset
(features_train, labels_train), (features_test, labels_test) = load_data()
_, rows, cols = features_train.shape
num_nodes = rows * cols
num_classes = len(np.unique(labels_train))

print('Number of training samples: %d' % features_train.shape[0])
print('Number of test samples: %d' % features_test.shape[0])
print('Rows: %d' % rows)
print('Columns: %d' % cols)
print('Number of classes: %d' % num_classes)

fig = plt.figure(figsize=(8, 3))
for i in range(num_classes):
    x = fig.add_subplot(2, 5, i + 1, xticks=[], yticks=[])
    features_idx = features_train[labels_train[:] == i, :]
    x.set_title("Digit: " + str(i))
    plt.imshow(features_idx[i], cmap='gray')
plt.show()


# Data preprocessing (converting to probabilistic values [0..1])
features_train = features_train.reshape(features_train.shape[0], num_nodes)
features_test = features_test.reshape(features_test.shape[0], num_nodes)
labels_train = utils.to_categorical(labels_train, num_classes)
labels_test = utils.to_categorical(labels_test, num_classes)

model = models.Sequential()
model.add(layers.Dense(100, input_dim=num_nodes))
model.add(layers.Activation('relu'))
model.add(layers.Dense(num_classes))
model.add(layers.Activation('softmax'))

sgd = optimizers.SGD(lr=0.01)
model.compile(optimizer=sgd, loss='mse', metrics=['accuracy'])
model.summary()

info = model.fit(features_train, labels_train, batch_size=64,
                 epochs=240, verbose=2, validation_split=0.3)

result = model.predict(features_test)
predicted_class = np.argmax(result, axis=1)
true_class = np.argmax(labels_test, axis=1)
num_correct = np.sum(predicted_class == true_class)
accuracy = float(num_correct) / result.shape[0]

# 96 % accuracy for activation='relu', batch=64, val_split=0.3 and epochs=240
print('Accuracy: %f.05' % accuracy)

fig, axs = plt.subplots(1, 2, figsize=(15, 5))
axs[0].plot(range(1, len(info.history['acc']) + 1), info.history['acc'])
axs[0].plot(range(1, len(info.history['val_acc']) + 1), info.history['val_acc'])
axs[0].set_title('Model Accuracy')
axs[0].set_ylabel('Accuracy')
axs[0].set_xlabel('Epoch')
axs[0].set_xticks(np.arange(1, len(info.history['acc']) + 1), len(info.history['acc']) / 10)
axs[0].legend(['train', 'val'], loc='best')

axs[1].plot(range(1, len(info.history['loss']) + 1), info.history['loss'])
axs[1].plot(range(1, len(info.history['val_loss']) + 1), info.history['val_loss'])
axs[1].set_title('Model Loss')
axs[1].set_ylabel('Loss')
axs[1].set_xlabel('Epoch')
axs[1].set_xticks(np.arange(1, len(info.history['loss']) + 1), len(info.history['loss']) / 10)
axs[1].legend(['train', 'val'], loc='best')
plt.show()
