from __future__ import print_function
import six.moves.cPickle as pickle
import gzip, os, sys, timeit, warnings
import numpy as np
from skimage import io, color, transform, img_as_ubyte
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
import random
import argparse

class DataPreProcess(object):
    def __init__(self, 
                 input_root_dir, 
                 output_root_dir, 
                 label_map, 
                 target_height, 
                 target_width):
        
        self.input_root_dir = input_root_dir
        self.output_root_dir = output_root_dir
        self.label_map = label_map
        self.target_height = target_height
        self.target_width = target_width
    
    
    def load_train_data(self, label_name, data_type):
        data_dir = os.path.join(self.output_root_dir, data_type + '_' + label_name + '.pkl')
        f = open(data_dir, 'rb')
        data = pickle.load(f)
        f.close()
        return data
    
    
    def load_all_train_data(self, data_type):
        images = []
        labels = []
        for label_name in self.label_map:
            data_dir = os.path.join(self.output_root_dir, data_type + '_' + label_name + '.pkl')
            f = open(data_dir, 'rb')
            curr_image, curr_label = pickle.load(f)
            f.close()
            images.extend(curr_image)
            labels.extend(curr_label)
        
        data = zip(images, labels)
        random.shuffle(data)
        images, labels = zip(*data)
        
        return images, labels
    
    
    def show_train_data(self, label_name):
        train_images, train_labels = self.load_train_data(label_name, 'train')
        # train_images, train_labels = self.load_all_train_data('train')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for image in train_images:
                plt.imshow(image)
#                 plt.imshow(image, cmap='Greys_r')
                plt.pause(2)
                

    def convert_images(self, rgb_flag, directory):
        images = []
        image_name_list = image_search(directory) # search images
        for image_name in image_name_list: # each files
            image_dir = os.path.join(directory, image_name)
            # if rgb_flag is true, preserve three channels
            if rgb_flag == True:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    image = io.imread(image_dir)
                    resized_image = resize_and_pad_zero(image, self.target_height, self.target_width)
                    padded_image = img_as_ubyte(resized_image)
                    image_shape = padded_image.shape
                    if(image_shape[2] != 3):
                        padded_image = np.delete(padded_image, 3, 2)
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    image = color.rgb2gray(io.imread(image_dir))
                    padded_image = img_as_ubyte(resize_and_pad_zero(image, self.target_height, self.target_width))
                    
            images.append((padded_image))
            
        return images
            

    def preprocess_test_data(self, rgb_flag):
        converted_images = self.convert_images(rgb_flag, self.input_root_dir)
        images = np.stack(converted_images, axis=0)
        train_output_file_name = 'test_data.pkl'
        write_data(os.path.join(self.output_root_dir, train_output_file_name), images)
        

    def preprocess_train_data(self, rgb_flag):
        for folder_name in self.label_map:
            print('writing files for class ' + folder_name)
            sub_dir = os.path.join(self.input_root_dir, folder_name)
            current_label = self.label_map[folder_name]
            images = self.convert_images(rgb_flag, sub_dir)            
            # separate data into train validation test sets, percentage: 0.8, 0.1, 0.1
            images_train, images_vali_test = train_test_split(images, test_size=0.2, random_state=42)
            images_validation, images_test = train_test_split(images_vali_test, test_size=0.5, random_state=35)
            labels_train = np.full((len(images_train),), current_label, dtype=np.int)
            labels_validation = np.full((len(images_validation),), current_label, dtype=np.int)
            labels_test = np.full((len(images_test),), current_label, dtype=np.int)
            # use pickle to write back files
            train_output_file_name = 'train' + '_' + folder_name + '.pkl'
            validation_output_file_name = 'validation' + '_' + folder_name + '.pkl'
            test_output_file_name = 'test' + '_' + folder_name + '.pkl'
            write_data(os.path.join(self.output_root_dir, train_output_file_name), [images_train, labels_train])
            write_data(os.path.join(self.output_root_dir, validation_output_file_name), [images_validation, labels_validation])
            write_data(os.path.join(self.output_root_dir, test_output_file_name), [images_test, labels_test])
            

def write_data(directory, data):
    output_file = open(directory, 'wb')
    pickle.dump(data, output_file, -1)
    output_file.close()

def image_search(root_dir):
    file_names = []
    image_file_names = []
    for root, sub_dir, current_file_names in os.walk(root_dir):
        file_names.extend(current_file_names)
        
    for file_name in file_names:
        ext = os.path.splitext(file_name)[1].lower()
        if ext in ('.jpg', '.jpeg', '.png', '.bmp', '.gif'):
            image_file_names.append(file_name)
            
    return image_file_names


def get_padding(target_length, input_length):
    if(input_length >= target_length):
        return (0, 0)
    pad_1 = (target_length - input_length) / 2
    pad_2 = target_length - input_length - pad_1
    return (pad_1, pad_2)


def resize_and_pad_zero(image, target_height, target_width):
    # determine a resize ratio
    image_shape = image.shape
    original_height = image_shape[0]
    original_width = image_shape[1]
    height_ratio = target_height / float(original_height)
    width_ratio = target_width / float(original_width)
    # choose the smaller ratio, so that we can pad rather than crop
    if(height_ratio <= width_ratio):
        ratio = height_ratio
    else:
        ratio = width_ratio
    # resize    
    resized_image = transform.rescale(image, ratio)
    resized_image_shape = resized_image.shape
    resized_height = resized_image_shape[0]
    resized_width = resized_image_shape[1]
    # pad image with 0
    pad_top, pad_bottom = get_padding(target_height, resized_height)
    pad_left, pad_right = get_padding(target_width, resized_width)
    
    padded_image = np.lib.pad(resized_image, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), 'constant')
    # print("ratio %f, padded height %d, padded width %d" % (ratio, padded_image.shape[0], padded_image.shape[1]))
    return padded_image
    

if __name__ == '__main__':
    
    label_map = {'guinea pig': 0, 
                 'squirrel': 1,
                 'sikadeer': 2,
                 'fox': 3,
                 'dog': 4,
                 'wolf': 5,
                 'cat': 6, 
                 'chipmunk': 7,
                 'giraffe': 8, 
                 'reindeer': 9,
                 'hyena': 10,
                 'weasel': 11 
                 }
    
    parser = argparse.ArgumentParser(prog='data_preprocessing')
    parser.add_argument('-m', '--mode', help="data process mode")
    parser.add_argument("-i", "--image", help="image directory", default="")
    parser.add_argument("-d", "--data", help="output data directory", default="")
    parser.add_argument("-r", "--rgb", type=bool, help="rgb flag", default="")
    parser.add_argument("--height", type=int, help="output data directory", default="")
    parser.add_argument("--width", type=int, help="output data directory", default="")
    
    args = parser.parse_args()
    
    image_root_dir = args.image
    pickle_root_dir = args.data
    image_height = args.height
    image_width = args.width
    rgb_flag = args.rgb
    
    data_preprocessor = DataPreProcess(image_root_dir, pickle_root_dir, label_map, target_height=image_height, target_width=image_width)
    if(args.mode == "train"):
        data_preprocessor.preprocess_train_data(rgb_flag)
    elif(args.mode == "test"):
        data_preprocessor.preprocess_test_data(rgb_flag)
    elif(args.mode == "show"):
        data_preprocessor.show_train_data('sikadeer')
