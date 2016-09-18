from __future__ import print_function
import numpy as np
np.random.seed(1234)

import argparse
import os
import six.moves.cPickle as pickle
import matplotlib.pyplot as plt
from keras.models import load_model
from image_preprocess import ImagePreprocess
from image_preprocess import image_search
from vgg_net import vgg_net_11
from vgg_net import vgg_net_11_temp


def load_test_data(data_dir, n_partition, partition_idx, avg_rgb):
    f = open(data_dir, 'rb')
    data, names = pickle.load(f)
    f.close()
    
    data_len = data.shape[0]
    partition_len = data_len / n_partition
    data = data[partition_idx * partition_len : (partition_idx + 1) * partition_len,]
    names = names[partition_idx * partition_len : (partition_idx + 1) * partition_len]
    data = (data.astype('float32') - avg_rgb)
    data = np.divide(data, 255.0)
    
    test_data = np.rollaxis(data, 3, 1)
    return test_data, names

# def load_validation_data(data_dir):
#     f = open(data_dir, 'rb')
#     data = pickle.load(f)
#     f.close()
#     test_data = np.rollaxis(data, 3, 1)
#     return test_data

def write_result_file(result):
    result_file = open('result_file.txt', 'w')
    for image_name, best_label, best_prob, second_label, second_prob in result:
        result_file.write("%s\t%d\t%.6f\t%d\t%.6f" % (image_name, best_label, best_prob, second_label, second_prob))
        result_file.write('\n')


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(prog='vgg_net')
    parser.add_argument("-d", "--data", help="input data directory", default="")
    parser.add_argument("-i", "--image", help="input image directory", default="")
    parser.add_argument("-w", "--weight", help="network weight directory", default="")
    
    args = parser.parse_args()
    
    data_dir = args.data
    image_dir = args.image
    weight_dir = args.weight
    
    image_preprocess = ImagePreprocess()
    
    n_partition = 4
    avg_rgb = np.asarray([132.4509, 123.5161, 105.4855], dtype=float)
    
#     image_name_list = image_search(image_dir)
#     for idx in range(len(image_name_list)):
#         image_name_list[idx] = os.path.splitext(image_name_list[idx])[0]
        
    vgg_11 = vgg_net_11_temp()
    vgg_11.load_weights(weight_dir)
    
    results = []
    full_idx = 0
    for partition_idx in range(n_partition):
        print('predict for partition %d' % (partition_idx))
        test_data, image_names = load_test_data(data_dir, n_partition, partition_idx, avg_rgb)    
        test_probability = vgg_11.predict_proba(test_data)
        
        for curr_partition_idx in range(len(test_data)):
            curr_data_probillity = test_probability[curr_partition_idx,:]
            top_2_labels = curr_data_probillity.argsort()[-2:][::-1]
            best_label = top_2_labels[0]
            second_label = top_2_labels[1]
            best_probability = curr_data_probillity[best_label]
            second_probability = curr_data_probillity[second_label]
            results.append([image_names[curr_partition_idx], best_label, best_probability, second_label, second_probability])
            full_idx = full_idx + 1
          
    write_result_file(results)  
#             print(image_preprocess.reverse_label_map[best_label])
#             plt.title(image_preprocess.reverse_label_map[best_label])
#             plt.imshow(np.rollaxis(test_data[curr_partition_idx,:], 0, 3))
#             plt.pause(3)  
#     test_data = load_test_data(data_dir)
    
        
    
    
    
    
    
    
    
    
    #write_result_file(results)
    
    

