# Step 1: Load the required libraries
import os
import cv2
import numpy as np
from keras.utils import to_categorical
from keras.layers import MaxPooling2D, Dense, Dropout, Activation, Flatten, Conv2D
from keras.models import Sequential, model_from_json
import pickle
from sklearn.model_selection import train_test_split

# Step 2: Load the dataset
path = 'Dataset'
labels = []
X = []
Y = []

def getID(name):
    if name in labels:
        return labels.index(name)
    labels.append(name)
    return len(labels) - 1

# Identify labels
for root, dirs, files in os.walk(path):
    for file in files:
        name = os.path.basename(root)
        if name not in labels:
            labels.append(name)

print("Labels:", labels)

# Step 3: Preprocess the dataset
for root, dirs, files in os.walk(path):
    for file in files:
        name = os.path.basename(root)
        filepath = os.path.join(root, file)
        if 'Thumbs.db' not in file:
            try:
                img = cv2.imread(filepath)
                img = cv2.resize(img, (32, 32))
                X.append(img)
                Y.append(getID(name))
                print(f"Loaded: {name} -> {filepath}")
            except Exception as e:
                print(f"Error loading image {filepath}: {e}")

X = np.asarray(X)
Y = np.asarray(Y)

# Save raw data
os.makedirs('model', exist_ok=True)
np.save('model/X.npy', X)
np.save('model/Y.npy', Y)

# Reload and normalize data
X = np.load('model/X.npy').astype('float32') / 255.0
Y = np.load('model/Y.npy')

# Visual check
cv2.imshow("Sample Image", X[3])
cv2.waitKey(0)
cv2.destroyAllWindows()

# Shuffle and one-hot encode
indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
Y = Y[indices]
Y = to_categorical(Y)

# Step 4: Dataset splitting
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

# Step 5: Load or build the model
model_path = 'model/model.json'
weights_path = 'model/model.weights.h5'  # ← Updated extension
history_path = 'model/history.pckl'

if os.path.exists(model_path) and os.path.exists(weights_path):
    with open(model_path, "r") as json_file:
        loaded_model_json = json_file.read()
        classifier = model_from_json(loaded_model_json)
    classifier.load_weights(weights_path)
    print(classifier.summary())

    with open(history_path, 'rb') as f:
        data = pickle.load(f)
    accuracy = data['accuracy'][9] * 100
    print(f"Training Model Accuracy = {accuracy:.2f}%")
else:
    classifier = Sequential()
    classifier.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(Flatten())
    classifier.add(Dense(units=256, activation='relu'))
    classifier.add(Dense(units=y_train.shape[1], activation='softmax'))

    classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print(classifier.summary())

    hist = classifier.fit(X_train, y_train, batch_size=8, epochs=20, shuffle=True,
                          verbose=2, validation_data=(X_test, y_test))

    # Save model
    classifier.save_weights(weights_path)  # ← Updated filename
    with open(model_path, "w") as json_file:
        json_file.write(classifier.to_json())
    with open(history_path, 'wb') as f:
        pickle.dump(hist.history, f)

    accuracy = hist.history['accuracy'][9] * 100
    print(f"Training Model Accuracy = {accuracy:.2f}%")