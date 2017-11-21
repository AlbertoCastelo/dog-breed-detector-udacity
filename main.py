

from sklearn.datasets import load_files
from keras.utils import np_utils
import numpy as np
from extract_bottleneck_features import extract_Resnet50
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from keras.preprocessing import image
from tqdm import tqdm
from PIL import ImageFile
from glob import glob
import random
import cv2
import matplotlib.pyplot as plt

# define function to load train, test, and validation datasets
def load_dataset(path):
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
    return dog_files, dog_targets


### TODO: Write a function that takes a path to an image as input
### and returns the dog breed that is predicted by the model.

class DogBreed():
    def path_to_tensor(self, img_path):
        # loads RGB image as PIL.Image.Image type
        img = image.load_img(img_path, target_size=(224, 224))
        # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
        x = image.img_to_array(img)
        # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
        return np.expand_dims(x, axis=0)

    def paths_to_tensor(self, img_paths):
        list_of_tensors = [self.path_to_tensor(img_path) for img_path in tqdm(img_paths)]
        return np.vstack(list_of_tensors)
    def load_model(self):
        ### TODO: Define your architecture.
        self.Resnet50_model = Sequential()

        self.Resnet50_model.add(GlobalAveragePooling2D(input_shape=(1, 1, 2048)))
        self.Resnet50_model.add(Dense(133, activation='softmax'))

        self.Resnet50_model.summary()
        ### TODO: Load the model weights with the best validation loss.
        self.Resnet50_model.load_weights('saved_models/weights.best.Resnet50.hdf5')

    def predict_dog_breed(self, img_path):
        tensor = self.path_to_tensor(img_path)
        print(tensor.shape)
        self.bottleneck_features = extract_Resnet50(tensor)
        print(self.bottleneck_features.shape)

        feat_expand = np.expand_dims(self.bottleneck_features, axis=0)
        print(feat_expand.shape)
        index = np.argmax(self.Resnet50_model.predict(self.bottleneck_features))

        return self.dog_names[index]

def main():
    dogbreed = DogBreed()
    dogbreed.load_model()
    dogbreed.predict_dog_breed("/home/acastelo/Escritorio/Udacity/dog-breed/dog-project/testing_images/example_2.jpg")


main()