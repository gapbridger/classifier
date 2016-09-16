from __future__ import print_function
import six.moves.cPickle as pickle
import gzip, os, sys, timeit, warnings
import numpy as np
from skimage import io, color, transform, img_as_ubyte
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
import random
import argparse

class ImagePreprocess(object):
    def __init__(self, 
                 input_root_dir, 
                 output_root_dir, 
                 label_map, 
                 target_scale): # target size of smallest side
        
        self.input_root_dir = input_root_dir
        self.output_root_dir = output_root_dir
        self.label_map = label_map
        self.target_scale = target_scale
    

    def scale_images(self, folder_directory):
        images = []
        image_name_list = image_search(folder_directory) # search images
        for image_name in image_name_list: # each files
            image_directory = os.path.join(folder_directory, image_name)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                image = io.imread(image_directory)
                image_shape = image.shape
                # get rid of the last channel
                if(image_shape[2] != 3):
                    image = np.delete(image, 3, 2)
                    
                original_height = image_shape[0]
                original_width = image_shape[1]
                height_ratio = self.target_scale / float(original_height)
                width_ratio = self.target_scale / float(original_width)
                # choose the larger ratio for crop
                if(height_ratio <= width_ratio):
                    ratio = width_ratio
                else:
                    ratio = height_ratio
                # resize
                resized_image = transform.rescale(image, ratio)
                resized_image_shape = resized_image.shape
                min_image_size = min(resized_image_shape[:2])
#                 print('minimum size of the image is %d...' % (min_image_size))
                resized_image = img_as_ubyte(resized_image)
#                 plt.imshow(resized_image)
#                 plt.pause(3)
            images.append(resized_image)
            
        return images
    
    def crop_images(self, image_list):
        idx = 0
        for image in image_list: # each files
            image_height = image.shape[0]
            image_width = image.shape[1]
            crop_size = 224
            image = image[(image_height - crop_size) / 2 : (image_height + crop_size) / 2, 
                                  (image_width - crop_size) / 2 : (image_width + crop_size) / 2, :]
            image_list[idx] = image
#             plt.imshow(image_list[idx])
#             plt.pause(3)
            idx += 1
            
        return image_list
    
    def show_image(self, partition_index):
        data_dir = os.path.join(self.output_root_dir, ('train_cropped_224_224_list_%d.pkl') % (partition_index))
        f = open(data_dir, 'rb')
        train_images, train_labels = pickle.load(f)
        f.close()
        reverse_label_map = {0: 'guinea pig', 1: 'squirrel', 2: 'sikadeer', 3: 'fox', 4: 'dog', 5: 'wolf', 
                             6: 'cat', 7: 'chipmunk', 8: 'giraffe', 9: 'reindeer', 10: 'hyena', 11: 'weasel'}
        # train_images, train_labels = self.load_all_train_data('train')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for i in range(train_images.shape[0]):
                plt.title(reverse_label_map[train_labels[i]])
                plt.imshow(train_images[i,:,:,:])
                plt.pause(2)
    
    def persist_resized_train_image_list(self):
        for folder_name in self.label_map:
            print('writing files for class ' + folder_name)
            sub_dir = os.path.join(self.input_root_dir, folder_name)
            current_label = self.label_map[folder_name]
            curr_image_list = self.scale_images(sub_dir)
            # train validation split 0.9 / 0.1
            curr_train_image_list, curr_validation_image_list = train_test_split(curr_image_list, test_size=0.1, random_state=42)
            curr_train_label_list = np.full((len(curr_train_image_list),), current_label, dtype=np.int)
            curr_validation_label_list = np.full((len(curr_validation_image_list),), current_label, dtype=np.int)
            
            print('processed %d %s images' % (len(curr_image_list), folder_name))
            # use pickle to write back files
            write_data(os.path.join(self.output_root_dir, 'train_images_labels_' + folder_name + '.pkl'), [curr_train_image_list, curr_train_label_list])
            write_data(os.path.join(self.output_root_dir, 'validation_images_labels_' + folder_name + '.pkl'), [curr_validation_image_list, curr_validation_label_list])
        
        
    def persist_cropped_train_image_list(self, partition_index, n_partition):
        
        full_train_cropped_image_list = []
        full_train_label_list = []
        for folder_name in self.label_map:
            
            data_dir = os.path.join(self.input_root_dir, 'train_images_labels_' + folder_name + '.pkl')
            f = open(data_dir, 'rb')
            curr_train_image_list, curr_train_label_list = pickle.load(f)
            f.close()
            
            batch_length = len(curr_train_image_list) / n_partition
            
            if(partition_index < n_partition):
                curr_train_image_list = self.crop_images(curr_train_image_list[(partition_index - 1) * batch_length : partition_index * batch_length])
                curr_train_label_list = curr_train_label_list[(partition_index - 1) * batch_length : partition_index * batch_length]
            else:
                curr_train_image_list = self.crop_images(curr_train_image_list[(partition_index - 1) * batch_length:])
                curr_train_label_list = curr_train_label_list[(partition_index - 1) * batch_length:]
            
            full_train_cropped_image_list.extend(curr_train_image_list)
            full_train_label_list.extend(curr_train_label_list)
            
            print('processed %d %s images' % (len(curr_train_image_list), folder_name))
            print('total image list length %d' % (len(full_train_cropped_image_list)))
            
        data = zip(full_train_cropped_image_list, full_train_label_list)
        random.shuffle(data)
        full_train_cropped_image_list, full_train_label_list = zip(*data)
        full_train_cropped_image_list = np.stack(full_train_cropped_image_list, axis=0)
        full_train_label_list = np.asarray(full_train_label_list, dtype=np.int)
        print('saving pickle..')
        write_data(os.path.join(self.output_root_dir, ('train_cropped_224_224_list_%d.pkl') % (partition_index)), [full_train_cropped_image_list, full_train_label_list])



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


if __name__ == '__main__':
    
    label_map = {'guinea pig': 0, 'squirrel': 1, 'sikadeer': 2, 'fox': 3, 'dog': 4, 'wolf': 5,
                 'cat': 6, 'chipmunk': 7, 'giraffe': 8, 'reindeer': 9, 'hyena': 10, 'weasel': 11}
    
    
    
    parser = argparse.ArgumentParser(prog='data_preprocessing')
    parser.add_argument('-m', '--mode', help="data process mode")
    parser.add_argument("-i", "--input", help="input data directory", default="")
    parser.add_argument("-o", "--output", help="output data directory", default="")
    parser.add_argument("-s", "--scale", type=int, help="scale", default=256)
    
    args = parser.parse_args()
    
    input_root_dir = args.input
    output_root_dir = args.output
    target_scale = args.scale
    
    data_preprocessor = ImagePreprocess(input_root_dir, output_root_dir, label_map, target_scale=target_scale)
    if(args.mode == "resize"):
        data_preprocessor.persist_resized_train_image_list()
    elif(args.mode == "crop"):
        n_partition = 8
        for i in range(n_partition):
            print('partition %d' % (i + 1))
            data_preprocessor.persist_cropped_train_image_list(i + 1, n_partition)
    elif(args.mode == "show"):
        data_preprocessor.show_image(1)
