# ===============
# Color Changing
# ===============
import os
import numpy as np
import matplotlib.pyplot as plt
import colorsys
from PIL import Image


def rewrite_color(lightness, color_list):
    return color_list[int(np.log(lightness+1) - 1)] 


os.chdir("/home/leshan/Datalab/tests")
im = Image.open('test.jpg') # Can be many different formats.
pix = im.load()

length, width = im.size
color_list = [(255, 205, 178), (255, 180, 162), (229, 152, 155), (181, 131, 141), (109, 104, 117)][::-1]

vibration, new_pixels = [[], []]
for x in range(length):
    for y in range(width):
        r, g, b = pix[x, y]
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        vibration.append(v)
        pix[x,y] = rewrite_color(v, color_list)
im.show()
plt.imshow(np.reshape(vibration, (length, width)).T, cmap='hot', interpolation='nearest')


# ===============
# Test Keras
# ===============


#import numpy as np
#np.random.seed(123)  # for reproducibility
#
#from keras.models import Sequential
#from keras.layers import Dense, Dropout, Activation, Flatten
#from keras.layers import Convolution2D, MaxPooling2D
#from keras.utils import np_utils
#
#from keras.datasets import mnist
# 
## 将MNIST 数据加载为训练集和测试集
#(X_train, y_train), (X_test, y_test) = mnist.load_data()
#
#from matplotlib import pyplot as plt
#plt.imshow(X_train[0])
#
#X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
#X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
#
#X_train = X_train.astype('float32')
#X_test = X_test.astype('float32')
#X_train /= 255
#X_test /= 255
#
#
#model = Sequential()
#
#model.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(1,28,28)))

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten

model = Sequential([
    Dense(32, input_shape=(784,)),
    Activation('relu'),
    Dense(10),
    Activation('softmax'),
])




# ===============
# Test Mido
# ===============

from mido import Message, MidiFile, MidiTrack

# Change Current Directory
import sys
import os
print(os.getcwd())
os.chdir('/home/leshan/Datalab/tests/')

#type 0 (single track)
#type 1 (synchronous start)
#type 2 (asynchronous start)
mid = MidiFile(type=1)
track = MidiTrack()
mid.tracks.append(track)


"""
# Loop for generating Note
# Loop times depends on chorus length

# Pay attention to time/duration
"""
#Note 64 = E3 = 12notes*5oct + 5
track.append(Message('program_change', program=12, time=0))
track.append(Message('note_on', note=64, velocity=64, time=256))
#track.append(Message('note_off', note=64, velocity=127, time=32))
mid.save('amidi.mid')
