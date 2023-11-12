import os
import glob
import numpy as np
from PIL import Image
from skimage.morphology import binary_closing, binary_opening, erosion
from scipy import ndimage
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.resnet50 import ResNet50
from sklearn.model_selection import train_test_split

def resize_images(input_dir, output_dir, basewidth=300):
    files = glob.glob(input_dir)
    for file in files:
        img = Image.open(file)
        wpercent = (basewidth / float(img.size[0]))
        hsize = int((float(img.size[1]) * float(wpercent)))
        img = img.resize((basewidth, hsize), Image.ANTIALIAS)
        file_save = file.replace('/Labelled\\', '/Resized\\')
        img.save(file_save)

def process_images(files, selem, cr, base_dir):
    files_bgremoved = [file.replace('/Labelled\\', '/BGRemoved\\') for file in files]
    idx = 0
    for file, file_save in zip(files, files_bgremoved):
        bg_frac = 0
        thres = 220
        img = Image.open(file)
        im_arr = np.array(img)
        R = im_arr[:, :, 0]
        G = im_arr[:, :, 1]
        B = im_arr[:, :, 2]

        while bg_frac < 0.6:
            bg_mask = ((R > thres) | (B > thres))
            bg_frac = bg_mask.sum() / len(bg_mask.flatten())
            thres -= 5

        bg_mask = binary_closing(erosion(binary_opening(bg_mask, selem), np.ones((3, 3)), np.ones((5, 5))))
        label, num_label = ndimage.label(~bg_mask)
        size = np.bincount(label.ravel())
        biggest_label = size[1:].argmax() + 1
        bg_mask = label == biggest_label
        im_arr[~bg_mask, 0] = 255
        im_arr[~bg_mask, 1] = 255
        im_arr[~bg_mask, 2] = 255
        img = Image.fromarray(im_arr)
        img.save(file_save)
        idx += 1

def train_vgg16(base_dir, batch_size=32, EPOCHS=30):
    # Rest of your VGG16 model training code here
    pass

def train_vgg19(base_dir, batch_size=32, EPOCHS=30):
    # Rest of your VGG19 model training code here
    pass

def train_xception(base_dir, batch_size=32, EPOCHS=30):
    # Rest of your Xception model training code here
    pass

def train_resnet50(base_dir, batch_size=32, EPOCHS=30):
    # Rest of your ResNet50 model training code here
    pass

def load_and_train_custom_model(directory_root, EPOCHS=70, INIT_LR=1e-3, BS=16):
    # Rest of your custom model training code here
    pass

if __name__ == "__main__":
    input_dir = 'C:/Users/red/Downloads/archive/model/Labelled/*/*'
    output_dir = 'C:/Users/red/Downloads/archive/model/Resized'
    selem = np.zeros((25, 25))
    ci, cj = 12, 12
    cr = 13
    files = glob.glob(input_dir)
    resize_images(input_dir, output_dir)
    process_images(files, selem, cr, output_dir)  # Updated to use output_dir instead of base_dir
    train_vgg16(output_dir, batch_size=32, EPOCHS=30)
    train_vgg19(output_dir, batch_size=32, EPOCHS=30)
    train_xception(output_dir, batch_size=32, EPOCHS=30)
    train_resnet50(output_dir, batch_size=32, EPOCHS=30)
