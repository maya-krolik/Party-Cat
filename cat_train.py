from keras.models import Sequential
import keras.layers as layers
import keras.losses as losses
import keras.optimizers as optimizers
import numpy as np

# starting image size
image_size = 258

mode = 'lmks' 
output_size = 19
# number of classes classifying (x and y for each of the 9 landmarks for a total of 18)
# the unaccounted one describes how many data points are in each file (could have been removed, and probably should have)

# load data from dataset
data_00 = np.load('dataset/lmks_CAT_00.npy', allow_pickle=True)
data_01 = np.load('dataset/lmks_CAT_01.npy', allow_pickle=True)
data_02 = np.load('dataset/lmks_CAT_02.npy', allow_pickle=True)
print("\nLoading Data complete\n")
# data_03 = np.load('dataset/lmks_CAT_03.npy', allow_pickle=True)
# data_04 = np.load('dataset/lmks_CAT_04.npy', allow_pickle=True)
# data_05 = np.load('dataset/lmks_CAT_05.npy', allow_pickle=True)
# data_06 = np.load('dataset/lmks_CAT_06.npy', allow_pickle=True)

# divide into train and test data
# partial data train
x_train = np.concatenate((data_00.item().get('imgs'), data_01.item().get('imgs')))
y_train = np.concatenate((data_00.item().get(mode), data_01.item().get(mode)))

# full data train
# x_train = np.concatenate((data_00.item().get('imgs'), data_01.item().get('imgs'), data_02.item().get('imgs'), data_03.item().get('imgs'), data_04.item().get('imgs'), data_05.item().get('imgs')), axis=0)
# y_train = np.concatenate((data_00.item().get(mode), data_01.item().get(mode), data_02.item().get(mode), data_03.item().get(mode), data_04.item().get(mode), data_05.item().get(mode)), axis=0)

# partial data test
x_test = np.array(data_02.item().get('imgs'))
y_test = np.array(data_02.item().get(mode))

# full data test
# x_test = np.array(data_06.item().get('imgs'))
# y_test = np.array(data_06.item().get(mode))

# reshape train and test data into 
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (-1, image_size, image_size, 3))
x_test = np.reshape(x_test, (-1, image_size, image_size, 3))

y_train = np.reshape(y_train, (-1, output_size))
y_test = np.reshape(y_test, (-1, output_size))

print("\nTraining time :)\n")

class Net():
    def __init__(self, input_shape):
        self.model = Sequential()

        # (size - kernal)/ stride + padding + 1
        # (258 - 13)/ 5 = 49 + 1 = 50
        self.model.add(layers.Conv2D(
            8, # filters
            13, # kernals
            strides = 5, # step size
            activation = "relu",
            input_shape = input_shape,
        )) # output 50 x 50 x 8

        self.model.add(layers.MaxPool2D(pool_size = 2))
        # output 25 x 25 x 8

        # (25 - 3)/1 = 22 + 1 = 23
        self.model.add(layers.Conv2D(
            8, # filters
            3, # kernal
            strides = 1,
            activation = "relu"
        )) # output 23 x 23 x 8

        self.model.add(layers.ZeroPadding2D(
             padding = ((1,0), (1,0)),
         ))# output 24 x 24 x 8

        self.model.add(layers.MaxPool2D(pool_size = 2))
        # output 12 x 12 x 8

        self.model.add(layers.Flatten())
        # output 1152

        self.model.add(layers.Dense(256, activation = "relu"))
        self.model.add(layers.Dense(64, activation = "relu"))

        if mode == "bbs":
            self.model.add(layers.Dense(16, activation = "relu"))
            self.model.add(layers.Dense(4, activation = "softmax"))
        else:
            self.model.add(layers.Dense(19, activation = "relu"))

        self.loss = losses.CategoricalCrossentropy()
        self.optimizer = optimizers.SGD(learning_rate = 0.0001)
        self.model.compile(
            loss = self.loss,
            optimizer = self.optimizer, 
            metrics = ["accuracy"],
        )
    
    def __str__(self):
        self.model.summary()
        return ""
    
net = Net((image_size, image_size, 3))
print(net)

net.model.fit(
    x_train, 
    y_train,
    batch_size = 32,
    epochs = 40,
    verbose = 2,
    validation_data = (x_test, y_test),
    validation_batch_size = 32,
)

net.model.save("kat_model_features_0")