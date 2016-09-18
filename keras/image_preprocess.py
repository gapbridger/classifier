from __future__ import print_function
import six.moves.cPickle as pickle
import gzip, os, sys, timeit, warnings
import numpy as np
from skimage import io, color, transform, img_as_ubyte
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
import random
import argparse
import gc

class ImagePreprocess(object):
    def __init__(self, 
                 input_root_dir = '', 
                 output_root_dir = '', 
                 target_scale = 256): # target size of smallest side
        
        self.input_root_dir = input_root_dir
        self.output_root_dir = output_root_dir
        self.target_scale = target_scale
        self.label_map = {'guinea pig': 0, 'squirrel': 1, 'sikadeer': 2, 'fox': 3, 'dog': 4, 'wolf': 5,
                          'cat': 6, 'chipmunk': 7, 'giraffe': 8, 'reindeer': 9, 'hyena': 10, 'weasel': 11}
        self.reverse_label_map = {0: 'guinea pig', 1: 'squirrel', 2: 'sikadeer', 3: 'fox', 4: 'dog', 5: 'wolf', 
                                  6: 'cat', 7: 'chipmunk', 8: 'giraffe', 9: 'reindeer', 10: 'hyena', 11: 'weasel'}
    

    def scale_images(self, folder_directory, image_name_flag=False):
        images = []
        submit_image_names = []
        image_name_list = image_search(folder_directory) # search images
        
        
        for image_name in image_name_list: # each files
            image_directory = os.path.join(folder_directory, image_name)
            prefix = os.path.splitext(image_name)[0]
            submit_image_names.append(prefix)
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
            if(len(images) % 500 == 0):
                print('image list length %d' % (len(images)))
            
        if(image_name_flag):
            return images, submit_image_names
        else:
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
    

    def load_data_partition(self, partition_index, data_type):
        data_dir = os.path.join(self.input_root_dir, ('%s_cropped_224_224_ndarray_%d.pkl') % (data_type, partition_index))
        f = open(data_dir, 'rb')
        train_images, train_labels = pickle.load(f)
        f.close()
        return train_images, train_labels

    def show_image(self, partition_index, data_type):
        train_images, train_labels = self.load_data_partition(partition_index, data_type)
        
#         data_dir = '/home/tao/Projects/bot-match/test_data/data/test_set_4/test_data.pkl'
#         f = open(data_dir, 'rb')
#         train_images = pickle.load(f)
#         f.close()
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for i in range(train_images.shape[0]):
                plt.title(self.reverse_label_map[train_labels[i]])
                plt.imshow(train_images[i,:,:,:])
                plt.pause(2)
    
    def persist_resized_train_image_and_label(self):
        for folder_name in self.label_map:
            print('writing files for class ' + folder_name)
            sub_dir = os.path.join(self.input_root_dir, folder_name)
            current_label = self.label_map[folder_name]
            curr_image_list = self.scale_images(sub_dir)
            # train validation split 0.9 / 0.1
            curr_train_image_list, curr_validation_image_list = train_test_split(curr_image_list, test_size=0.1, random_state=42)
            curr_train_label_list = np.full((len(curr_train_image_list),), current_label, dtype=np.int)
            curr_validation_label_list = np.full((len(curr_validation_image_list),), current_label, dtype=np.int)
            
            print('train image length %d' % (len(curr_train_image_list)))
            print('validation image length %d' % (len(curr_validation_image_list)))
            print('processed %d %s images' % (len(curr_image_list), folder_name))
            # use pickle to write back files
            write_data(os.path.join(self.output_root_dir, 'train_images_labels_' + folder_name + '.pkl'), [curr_train_image_list, curr_train_label_list])
            write_data(os.path.join(self.output_root_dir, 'validation_images_labels_' + folder_name + '.pkl'), [curr_validation_image_list, curr_validation_label_list])
            
    def persist_resized_test_image(self):
        test_image_list = self.scale_images(self.input_root_dir)
        print('test image length %d' % (len(test_image_list)))
        # use pickle to write back files
        write_data(os.path.join(self.output_root_dir, 'test_image_list.pkl'), test_image_list)
        
        
    def persist_cropped_train_image_ndarray(self, n_partition, data_type):
        n_partition_per_batch = 4
        
        if(n_partition % n_partition_per_batch != 0):
            print("incorrect number of partition per batch... n_partition should mod n_partition_per_batch 0")
            exit(1)
            
        for batch_idx in range(n_partition / n_partition_per_batch):
            full_train_cropped_images = [[]  for i in range(n_partition_per_batch)]
            full_train_labels = [[]  for i in range(n_partition_per_batch)]
            
            for folder_name in self.label_map:
                data_dir = os.path.join(self.input_root_dir, data_type + '_images_labels_' + folder_name + '.pkl')
                f = open(data_dir, 'rb')
                train_images, train_labels = pickle.load(f)
                f.close()
                
                batch_length = len(train_images) / n_partition
                
                for curr_batch_partition_idx in range(n_partition_per_batch):
                    partition_idx = batch_idx * n_partition_per_batch + curr_batch_partition_idx
                    
                    if(partition_idx < n_partition):
                        curr_train_images = self.crop_images(train_images[partition_idx * batch_length : (partition_idx + 1) * batch_length])
                        curr_train_labels = train_labels[partition_idx * batch_length : (partition_idx + 1) * batch_length]
                    else:
                        curr_train_images = self.crop_images(train_images[partition_idx * batch_length:])
                        curr_train_labels = train_labels[partition_idx * batch_length:]
                
                    if(len(full_train_cropped_images[curr_batch_partition_idx]) == 0):
                        full_train_cropped_images[curr_batch_partition_idx] = curr_train_images
                        full_train_labels[curr_batch_partition_idx] = curr_train_labels.tolist()
                    else:
                        full_train_cropped_images[curr_batch_partition_idx].extend(curr_train_images)
                        full_train_labels[curr_batch_partition_idx].extend(curr_train_labels.tolist())
                
                    print('processed %d %s images, partition index %d' % (len(curr_train_images), folder_name, partition_idx))
                    print('total image list length %d partition index %d' % (len(full_train_cropped_images[curr_batch_partition_idx]), partition_idx))
            
            for curr_batch_partition_idx in range(n_partition_per_batch):   
                partition_idx = batch_idx * n_partition_per_batch + curr_batch_partition_idx
                data = zip(full_train_cropped_images[curr_batch_partition_idx], full_train_labels[curr_batch_partition_idx])
                random.shuffle(data)
                shuffled_full_train_cropped_images, shuffled_full_train_labels = zip(*data)
                shuffled_full_train_cropped_images = np.stack(shuffled_full_train_cropped_images, axis=0)
                shuffled_full_train_labels = np.asarray(shuffled_full_train_labels, dtype=np.int)
                print('saving pickle..')
                write_data(os.path.join(self.output_root_dir, ('%s_cropped_224_224_ndarray_%d.pkl') % (data_type, partition_idx)), [shuffled_full_train_cropped_images, shuffled_full_train_labels])
                
            del full_train_cropped_images
            del full_train_labels
            del curr_train_images
            del curr_train_labels
            del shuffled_full_train_cropped_images
            del shuffled_full_train_labels
            gc.collect()


    def mean_train_rgb(self, n_partition):
        rgb_sum = np.full((1, 1, 3), 0.0, dtype=np.float64)
        rgb_count = 0
        for partition_index in range(n_partition):
            train_images, train_labels = self.load_data_partition(partition_index, 'train')
            rgb_sum = rgb_sum + np.sum(train_images, axis=(0,1,2))
            rgb_count = rgb_count + train_images.shape[0] * train_images.shape[1] * train_images.shape[2]
                        
        rgb_avg = rgb_sum / float(rgb_count)
        
        return rgb_avg 
            
def read_cifar_10(cifar_root_directory):
    n_train_samples = 50000
    train_images = np.zeros((n_train_samples, 3, 32, 32), dtype="uint8")
    train_labels = np.zeros((n_train_samples,), dtype="uint8")

    for i in range(1, 6):
        data_path = os.path.join(cifar_root_directory, 'data_batch_' + str(i))
        data, labels = load_cifar_10_batch(data_path)
        train_images[(i - 1) * 10000: i * 10000, :, :, :] = data
        train_labels[(i - 1) * 10000: i * 10000] = labels
        
    data_path = os.path.join(cifar_root_directory, 'test_batch')
    test_images, test_labels = load_cifar_10_batch(data_path)
    
#     for i in range(100):
#         img = np.rollaxis(train_images[i,], 0, 3)
#         plt.imshow(img)
#         plt.pause(3)
    
    return train_images, train_labels, test_images, test_labels                
    
    
def load_cifar_10_batch(data_path):
    f = open(data_path, 'rb')
    d = pickle.load(f)
    
    for k, v in d.items():
        del(d[k])
        d[k.decode("utf8")] = v
    f.close()
    
    data = d["data"]
    labels = d["labels"]

    data = data.reshape(data.shape[0], 3, 32, 32)
    return data, labels
        

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
    
    parser = argparse.ArgumentParser(prog='data_preprocessing')
    parser.add_argument('-m', '--mode', help="data process mode")
    parser.add_argument("-i", "--input", help="input data directory", default="")
    parser.add_argument("-o", "--output", help="output data directory", default="")
    parser.add_argument("-s", "--scale", type=int, help="scale", default=256)
    parser.add_argument("-t", "--type", help="data type", default='validation')
    
    args = parser.parse_args()
    
    input_root_dir = args.input
    output_root_dir = args.output
    target_scale = args.scale
    data_type = args.type
    n_partition = 24
    
    data_preprocessor = ImagePreprocess(input_root_dir, output_root_dir, target_scale=target_scale)
    if(args.mode == "resize"):
        data_preprocessor.persist_resized_train_image_and_label()
    elif(args.mode == "crop"):
        data_preprocessor.persist_cropped_train_image_ndarray(n_partition, data_type)
    elif(args.mode == "show"):
        data_preprocessor.show_image(10, data_type)
    elif(args.mode == "mean"):
        rgb_avg = data_preprocessor.mean_train_rgb(n_partition)
        print('calculated average rgb value is: %.4f %.4f %.4f' % (rgb_avg[0, 0, 0], rgb_avg[0, 0, 1], rgb_avg[0, 0, 2]))
    elif(args.mode == "test"):
        test_image_list, test_image_names = data_preprocessor.scale_images(data_preprocessor.input_root_dir, True)
        print('test image length %d' % (len(test_image_list)))
        cropped_test_images = data_preprocessor.crop_images(test_image_list)
#         data_dir = os.path.join(data_preprocessor.input_root_dir,'test_image_list.pkl')
#         f = open(data_dir, 'rb')
#         test_images = pickle.load(f)
#         f.close()
#         cropped_test_images = data_preprocessor.crop_images(test_images)
        cropped_test_images = np.stack(cropped_test_images, axis=0)
        write_data(os.path.join(data_preprocessor.output_root_dir, 'test_data_with_name.pkl'), [cropped_test_images, test_image_names])