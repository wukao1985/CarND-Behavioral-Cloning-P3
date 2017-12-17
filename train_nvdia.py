import csv
import numpy as np
import matplotlib.image as mi

lines = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements=[]
for line in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = './data/IMG/' + filename
    rgb = mi.imread(current_path)
    images.append(rgb)
    measurement = float(line[3])
    measurements.append(measurement)

X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential, Model
from keras.layers import Input, Flatten, Dense, Activation, Conv2D, Convolution2D, MaxPooling2D
from keras.layers import Lambda, Cropping2D, Dropout


model = Sequential()
model.add(Lambda(lambda x: x/255.0-0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Conv2D(24,(5,5),strides = (2,2), activation = "relu"))
model.add(Conv2D(36,(5,5),strides = (2,2), activation = "relu"))
model.add(Conv2D(48,(5,5),strides = (2,2), activation = "relu"))
model.add(Conv2D(64,(3,3),activation = "relu"))
model.add(Conv2D(64,(3,3),activation = "relu"))
model.add(Flatten())
model.add(Dense(1164, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='tanh'))

model.summary()


model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=4)

model.save('model.h5')

