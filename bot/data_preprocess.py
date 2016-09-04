from __future__ import print_function
import six.moves.cPickle as pickle
import gzip, os, sys, timeit, warnings
import numpy as np
from skimage import io, color, transform, img_as_ubyte
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split

class DataPreProcess(object):
    def __init__(self, 
                 input_root_dir, 
                 output_root_dir, 
                 output_file_prefix, 
                 label_map, 
                 target_height, 
                 target_width):
        
        self.input_root_dir = input_root_dir
        self.output_root_dir = output_root_dir
        self.output_file_prefix = output_file_prefix
        self.label_map = label_map
        self.target_height = target_height
        self.target_width = target_width
    
    
    def load_data(self, label_name):
        data_dir = os.path.join(self.output_root_dir, self.output_file_prefix + label_name + '.pkl')
        f = open(data_dir, 'rb')
        train_data, validation_data, test_data = pickle.load(f)
        f.close()
        return train_data, validation_data, test_data
    
    
    def show_data(self, label_name):
        train_data, validation_data, test_data = self.load_data(label_name)
        train_images = train_data[0]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for image in train_images:
                plt.imshow(image, cmap='Greys_r')
                plt.pause(2)
                
                
    def preprocess_data(self):
        for folder_name in self.label_map:
            print('writing files for class ' + folder_name)
            sub_dir = os.path.join(self.input_root_dir, folder_name)
            current_label = self.label_map[folder_name]
            images = []
            image_name_list = image_search(sub_dir)   # search images
            for image_name in image_name_list:   # each files
                image_dir = os.path.join(sub_dir, image_name)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    image = img_as_ubyte(color.rgb2gray(io.imread(image_dir)))
                    padded_image = resize_and_pad_zero(image, self.target_height, self.target_width)
                images.append((padded_image))
            # separate data into train validation test sets, percentage: 0.8, 0.1, 0.1
            images_train, images_vali_test = train_test_split(images, test_size=0.2, random_state=42)
            images_validation, images_test = train_test_split(images_vali_test, test_size=0.5, random_state=35)
            labels_train = np.full((len(images_train),), current_label, dtype=np.int)
            labels_validation = np.full((len(images_validation),), current_label, dtype=np.int)
            labels_test = np.full((len(images_test),), current_label, dtype=np.int)
            # use pickle to write back files
            output_file_name = self.output_file_prefix + folder_name + '.pkl'
            output_file_dir = os.path.join(self.output_root_dir, output_file_name)
            output_file = open(output_file_dir, 'wb')
            pickle.dump([(images_train, labels_train), (images_validation, labels_validation), (images_test, labels_test)], output_file, -1)
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
    original_height, original_width = image.shape
    height_ratio = target_height / float(original_height)
    width_ratio = target_width / float(original_width)
    # choose the smaller ratio, so that we can pad rather than crop
    if(height_ratio <= width_ratio):
        ratio = height_ratio
    else:
        ratio = width_ratio
    # resize    
    resized_image = transform.rescale(image, ratio)
    resized_height, resized_width = resized_image.shape
    # pad image with 0
    pad_top, pad_bottom = get_padding(target_height, resized_height)
    pad_left, pad_right = get_padding(target_width, resized_width)
    
    padded_image = np.lib.pad(resized_image, ((pad_top, pad_bottom), (pad_left, pad_right)), 'constant')
    # print("ratio %f, padded height %d, padded width %d" % (ratio, padded_image.shape[0], padded_image.shape[1]))
    return padded_image
    
    



if __name__ == '__main__':
    
    label_map = {'cat': 0, 'chipmunk': 1, 'dog': 2, 'fox': 3, 'giraffe': 4, 'guinea pig': 5, 
                 'hyena': 6, 'reindeer': 7, 'sikadeer': 8, 'squirrel': 9, 'weasel': 10, 'wolf': 11}
    
    # uncomment following lines to fit your system settings
    # image_root_dir = 'your_image_root_directory'
    # pickle_root_dir = os.path.join(image_root_dir, 'pickle')
    data_preprocessor = DataPreProcess(image_root_dir, pickle_root_dir, 'labeled_raw_data_', label_map, target_height=180, target_width=240)
    # data_preprocessor.preprocess_data()
    # data_preprocessor.show_data('sikadeer')
