import numpy as np
import copy
import datetime
from scipy import stats
import torch
import torch.nn as nn

from vgg16 import VGG
from dataset import QDataset
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image

np.set_printoptions(suppress=True)


QUANTIZE_NUM = 127
INTERVAL_NUM = 2048
quantize_layer_lists = []

'''
Quantize part see nvidia ppt for more information
We use KL divergence to generate quantization tables
'''

class QuantizeLayer:
    def __init__(self, name,group_num):
        self.name = name
        self.group_num = group_num
        self.weight_scale = np.zeros(group_num)
        self.blob_max = 0.0
        self.blob_distubution_interval = 0.0
        self.blob_distubution = np.zeros(INTERVAL_NUM)
        self.blob_threshold = 0
        self.blob_scale = 1.0
        self.group_zero = np.zeros(group_num)

    def quantize_weight(self, weight_data):
        blob_group_data = np.array_split(weight_data, self.group_num)
        for i, group_data in enumerate(blob_group_data):
            max_val = np.max(group_data)
            min_val = np.min(group_data)
            threshold = max(abs(max_val), abs(min_val))
            if threshold < 0.0001:
                self.weight_scale[i] = 0
                self.group_zero[i] = 1
            else:
                self.weight_scale[i] = QUANTIZE_NUM / threshold
            print("%-20s group : %-5d max_val : %-10f scale_val : %-10f" % (self.name + "_param0", i, threshold, self.weight_scale[i]))

    def initial_blob_max(self, blob_data):
        # get the max value of blob
        max_val = np.max(blob_data)
        min_val = np.min(blob_data)
        self.blob_max = max(self.blob_max, max(abs(max_val), abs(min_val)))

    def initial_blob_distubution_interval(self):
        self.blob_distubution_interval = self.blob_max / INTERVAL_NUM
        print("%-20s max_val : %-10.8f distribution_intervals : %-10.8f" % (self.name, self.blob_max, self.blob_distubution_interval))

    def initial_histograms(self, blob_data):
        th = self.blob_max
        hist, hist_edge = np.histogram(blob_data, bins=INTERVAL_NUM, range=(0, th))
        self.blob_distubution += hist

    def quantize_blob(self):
        '''now we get 2048bins all input histogram'''
        distribution = np.array(self.blob_distubution)

        '''use KL divergence compute thireshold_bin'''
        threshold_bin = threshold_distribution(distribution) 
        self.blob_threshold = threshold_bin

        '''nvidia PPT: (threshold_bin+0.5) * interval =threshold'''
        threshold = (threshold_bin + 0.5) * self.blob_distubution_interval

        '''blob_scale = 127 / threshold '''
        self.blob_scale = QUANTIZE_NUM / threshold
        print("%-20s bin : %-8d threshold : %-10f interval : %-10f scale : %-10f" % (self.name, threshold_bin, threshold, self.blob_distubution_interval, self.blob_scale))


    
    
def threshold_distribution(distribution, target_bin=128):

    '''boundary question'''
    distribution = distribution[1:]
    length = distribution.size  #2047

    '''boundary sum'''
    threshold_sum = sum(distribution[target_bin:])
    kl_divergence = np.zeros(length - target_bin) #2047-128

    for threshold in range(target_bin, length):  #128 to 2046
        sliced_nd_hist = copy.deepcopy(distribution[:threshold])
        p = sliced_nd_hist.copy()
        '''boundary sum'''
        p[threshold-1] += threshold_sum
        '''next iterator boundary sum'''
        threshold_sum = threshold_sum - distribution[threshold]
        '''generate non zero array like [1,1,1,0,1,1,0] for norm part'''
        is_nonzeros = (p != 0).astype(np.int64)
        quantized_bins = np.zeros(target_bin, dtype=np.int64) #128
        '''compute num bins when merge'''
        num_merged_bins = sliced_nd_hist.size // target_bin
        '''merge sliced_nd_hist into 127bins'''
        for j in range(target_bin):
            start = j * num_merged_bins
            stop = start + num_merged_bins
            quantized_bins[j] = sliced_nd_hist[start:stop].sum()
        '''bounder sum'''
        quantized_bins[-1] += sliced_nd_hist[target_bin * num_merged_bins:].sum()

        '''use quantized_bins generate q(reconstruct p) distribution '''
        q = np.zeros(sliced_nd_hist.size, dtype=np.float64)
        for j in range(target_bin):
            start = j * num_merged_bins
            if j == target_bin - 1:
                stop = -1
            else:
                stop = start + num_merged_bins
            '''norm part  assert(sum q == 1) because it is probability distribution definition'''
            norm = is_nonzeros[start:stop].sum()
            if norm != 0:
                q[start:stop] = float(quantized_bins[j]) / float(norm)
        '''zero problem: 0 cannot be denominator'''
        q[p == 0] = 0
        p[p == 0] = 0.0001
        q[q == 0] = 0.0001
        '''from 0 record p,q kl value'''
        kl_divergence[threshold - target_bin] = stats.entropy(p, q)
    '''min index(threshold)'''
    min_kl_divergence = np.argmin(kl_divergence)
    threshold_value = min_kl_divergence + target_bin
    return threshold_value



def net_forward(net, image):
    net(image)


def weight_quantize(net):
    print("\nQuantize the kernel weight:")

    for name, layer in net.named_modules():
        '''only Q conv layer'''
        if isinstance(layer, nn.Conv2d):
            weight_blob = layer.weight.cpu().detach().numpy()
            '''record layername and conv weights channels'''
            quanitze_layer = QuantizeLayer(name,layer.out_channels)
            '''q weights: weight_scale = 127/max(abs(w_max),abs(w_min)) for every channel'''
            quanitze_layer.quantize_weight(weight_blob)
            '''append every Q layer object(class) to list in order to compute it's input Q'''
            quantize_layer_lists.append(quanitze_layer)
    return None                


class Hook_struct:
    def __init__(self, name, hook):
        self.name = name     
        self.hook = hook

hook_list = []
input_list = []  


'''collect layer's input and append to list'''
def get_feature(modules, input):
    input_list.append(input[0].cpu().detach().numpy().flatten())


def activation_quantize(net, images_files):
    print("\nQuantize the Activation:")
    print("image num:%d" % len(images_files))
    for name, layer in net.named_modules():
        if isinstance(layer, nn.Conv2d):
            '''register hook to collect conv layer's input'''
            hook = layer.register_forward_pre_hook(get_feature)
            '''hook struct to save'''
            hook_status = Hook_struct(name, hook)
            '''append hook struct to list'''
            hook_list.append(hook_status)


    with torch.no_grad():
     '''collect input for every conv layer'''
     for i, (image,label) in enumerate(images_files):
        image=image.cuda()
        net(image)
        for j, layer in enumerate(quantize_layer_lists):
            '''iterate every conv layer's input'''
            blob = input_list[j]
            '''record max blob = max(abs(data_max),abs(data_min))'''
            layer.initial_blob_max(blob)
        if i % 100 == 0:
            print("loop stage 1 : %d/%d" % (i, len(images_files)))

    '''interval = this_layer's blob_max / 2048'''
    for layer in quantize_layer_lists:
        layer.initial_blob_distubution_interval()

    '''clear'''
    input_list.clear()


    print("\nCollect histograms of activations:")
    with torch.no_grad():
     for i, (image,label) in enumerate(images_files):
        image = image.cuda()
        net(image)
        for j, layer in enumerate(quantize_layer_lists):
            blob = input_list[j]
            '''Establishing Input Histogram'''
            layer.initial_histograms(blob)
            print("loop stage 2 : %d/%d" % (i, len(images_files)))          

    '''Q input_blob'''
    for layer in quantize_layer_lists:
        layer.quantize_blob()  

    '''clear'''
    for hook in hook_list:
        hook.hook.remove()

    return None


'''save to file'''
def save_calibration_file(calibration_path):
    calibration_file = open(calibration_path, 'w')
    save_temp = []
    for layer in quantize_layer_lists:
        save_string = layer.name + "_param_0"
        for i in range(layer.group_num):
            save_string = save_string + " " + str(layer.weight_scale[i])
        save_temp.append(save_string)

    for layer in quantize_layer_lists:
        save_string = layer.name + " " + str(layer.blob_scale)
        save_temp.append(save_string)

    for data in save_temp:
        calibration_file.write(data + "\n")

    calibration_file.close()
    save_temp_log = []
    calibration_file_log = open(calibration_path + ".log", 'w')
    for layer in quantize_layer_lists:
        save_string = layer.name + ": value range 0 - " + str(layer.blob_max) \
                                 + ", interval " + str(layer.blob_distubution_interval) \
                                 + ", interval num " + str(INTERVAL_NUM) \
                                 + ", threshold num " + str(layer.blob_threshold) + "\n" \
                                 + str(layer.blob_distubution.astype(dtype=np.int64))
        save_temp_log.append(save_string)

    for data in save_temp_log:
        calibration_file_log.write(data + "\n")


def main():

    '''prepare part:build model vgg-mini,load weights and Q-datasets'''
    model=VGG()
    model.load_state_dict(torch.load('./checkpoint/ckpt.pth')['net'])
    test_augmentation = transforms.Compose([
        transforms.Resize((112, 112), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_cifar10 = QDataset(transform=test_augmentation)
    test_loader = DataLoader(dataset=test_cifar10, batch_size=1, shuffle=False)
    model.eval()
    net=model.cuda()

    '''
    Quantitative part
    '''


    time_start = datetime.datetime.now()
    '''weight's Q'''
    weight_quantize(net)
    '''input blobs Q'''
    activation_quantize(net, test_loader)
    '''save to Q table'''
    save_calibration_file('./vgg_cifar10.table')

    time_end = datetime.datetime.now()
    print("\nPyTorch Int8 Calibration table is done, it's cost %s." % (time_end - time_start))
if __name__ == "__main__":
    main()
