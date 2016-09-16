from __future__ import print_function
import six.moves.cPickle as pickle
import gzip, os, sys, timeit, warnings
import numpy as np
from skimage import io, color, transform, img_as_ubyte
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
import random

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
    
    
    def load_data(self, label_name, data_type):
        data_dir = os.path.join(self.output_root_dir, data_type + '_' + label_name + '.pkl')
        f = open(data_dir, 'rb')
        data = pickle.load(f)
        f.close()
        return data
    
    
    def load_all_data(self, data_type):
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
    
    
    def show_data(self, label_name):
        train_images, train_labels = self.load_data(label_name, 'train')
        # train_images, train_labels = self.load_all_data('train')
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
                    image = color.rgb2gray(io.imread(image_dir))
                    padded_image = img_as_ubyte(resize_and_pad_zero(image, self.target_height, self.target_width))
                images.append((padded_image))
            # separate data into train validation test sets, percentage: 0.8, 0.1, 0.1
            images_train, images_vali_test = train_test_split(images, test_size=0.2, random_state=42)
            images_validation, images_test = train_test_split(images_vali_test, test_size=0.5, random_state=35)
            labels_train = np.full((len(images_train),), current_label, dtype=np.int)
            labels_validation = np.full((len(images_validation),), current_label, dtype=np.int)
            labels_test = np.full((len(images_test),), current_label, dtype=np.int)
            # use pickle to write back files
            write_data(self.output_root_dir, folder_name, 'train', images_train, labels_train)
            write_data(self.output_root_dir, folder_name, 'validation', images_validation, labels_validation)
            write_data(self.output_root_dir, folder_name, 'test', images_test, labels_test)
            

def write_data(root_dir, label_name, data_type, images, labels):
        output_file_name = data_type + '_' + label_name + '.pkl'
        output_file_dir = os.path.join(root_dir, output_file_name)
        output_file = open(output_file_dir, 'wb')
        pickle.dump([images, labels], output_file, -1)
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
    label_map = {'cat': 6, 'chipmunk': 7, 'dog': 4, 'fox': 3, 'giraffe': 8, 'guinea pig': 0,
                 'hyena': 10, 'reindeer': 9, 'sikadeer': 2, 'squirrel': 1, 'weasel': 11, 'wolf': 5}
    # label_map = {'chipmunk': 1}
    image_root_dir = 'C:\\zzdata\\bot_train'
    pickle_root_dir = 'C:\\zzdata\\bot_train\\pickle'
    
    data_preprocessor = DataPreProcess(image_root_dir, pickle_root_dir, label_map, target_height=224, target_width=224)
    # data_preprocessor.preprocess_data()
    data_preprocessor.show_data('chipmunk')
