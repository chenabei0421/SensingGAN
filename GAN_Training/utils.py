import os
import cv2
import skimage
import numpy as np
import torch
import math

import network

# ----------------------------------------
#                 Network
# ----------------------------------------
def create_generator(opt):
    # Initialize the network
    generator = network.Generator()
    if opt.load_gname == '':
        # Init the network
        network.weights_init(generator, init_type = opt.init_type, init_gain = opt.init_gain)
        print('Generator are created!')
    else:
        # Load a pre-trained network
        pretrained_net = torch.load(opt.load_gname)
        load_dict_g(generator, pretrained_net)
        print('Generator are loaded!')
    return generator

def create_discriminator(opt):
    # Initialize the network
    discriminator = network.Discriminator()
    if opt.load_gname == '':
        # Init the network
        network.weights_init(discriminator, init_type = opt.init_type, init_gain = opt.init_gain)
        print('Discriminator are created!')
    else:
        # Load a pre-trained network
        pretrained_dnet = torch.load(opt.load_dname)
        load_dict_d(discriminator, pretrained_dnet)
        print('Discriminaotr are loaded!')
    return discriminator
    
def load_dict_g(process_net, pretrained_net):
    # Get the dict from pre-trained network
    pretrained_dict = pretrained_net
    # Get the dict from processing network
    process_dict = process_net.state_dict()
    # Delete the extra keys of pretrained_dict that do not belong to process_dict
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in process_dict}
    # Update process_dict using pretrained_dict
    process_dict.update(pretrained_dict)
    # Load the updated dict to processing network
    process_net.load_state_dict(process_dict)
    return process_net

def load_dict_d(process_net, pretrained_net):
    # Get the dict from pre-trained network
    pretrained_dict = pretrained_net
    # Get the dict from processing network
    process_dict = process_net.state_dict()
    # Delete the extra keys of pretrained_dict that do not belong to process_dict
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in process_dict}
    # Update process_dict using pretrained_dict
    process_dict.update(pretrained_dict)
    # Load the updated dict to processing network
    process_net.load_state_dict(process_dict)
    return process_net

# ----------------------------------------
#    Validation and Sample at training
# ----------------------------------------
def save_sample_png(sample_folder, sample_name, img_list, name_list, pixel_max_cnt = 255, height = -1, width = -1):
    # Save image one-by-one
    for i in range(len(img_list)):
        img = img_list[i]
        # Recover normalization
        img = img * 255.0
        # Process img_copy and do not destroy the data of img
        #print(img.size())
        img_copy = img.clone().data.permute(0, 2, 3, 1).cpu().numpy()
        img_copy = np.clip(img_copy, 0, pixel_max_cnt)
        img_copy = img_copy.astype(np.uint8)[0, :, :, :]
        img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
        if (height != -1) and (width != -1):
            img_copy = cv2.resize(img_copy, (width, height))
        # Save to certain path
        save_img_name = sample_name + '_' + name_list[i] + '.png'
        save_img_path = os.path.join(sample_folder, save_img_name)
        cv2.imwrite(save_img_path, img_copy)
        
def save_RID_png(sample_folder, sample_name, img_list, name_list, pixel_max_cnt = 255, height = -1, width = -1):
    # Save image one-by-one
    for i in range(len(img_list)):
        img = img_list[i]
        # Recover normalization
        img = img * 255.0
        # Process img_copy and do not destroy the data of img
        #print(img.size())
        img_copy = img.clone().data.permute(0, 2, 3, 1).cpu().numpy()
        img_copy = np.clip(img_copy, 0, pixel_max_cnt)
        img_copy = img_copy.astype(np.uint8)[0, :, :, :]
        img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
        if (height != -1) and (width != -1):
            img_copy = cv2.resize(img_copy, (width, height))
        # Save to certain path
        save_img_name = sample_name
        save_img_path = os.path.join(sample_folder, save_img_name[0])
        cv2.imwrite(save_img_path, img_copy)
  
def save_one_sample_png(sample_folder, sample_name, img_list, name_list, pixel_max_cnt = 255, height = -1, width = -1):
    # Save image one-by-one
    img = img_list
    # Recover normalization
    img = img * 255.0
    # Process img_copy and do not destroy the data of img
    #print(img.size())
    img_copy = img.clone().data.permute(2, 1, 0).cpu().numpy()
    img_copy = np.clip(img_copy, 0, pixel_max_cnt)
    img_copy = img_copy.astype(np.uint8)[ :, :, :]
    img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
    if (height != -1) and (width != -1):
        img_copy = cv2.resize(img_copy, (width, height))
    # Save to certain path
    save_img_name = sample_name + '_' + name_list + '.png'
    save_img_path = os.path.join(sample_folder, save_img_name)
    cv2.imwrite(save_img_path, img_copy)

def save_sample_png_test(sample_folder, sample_name, img_list, name_list, pixel_max_cnt = 255):
    # Save image one-by-one
    for i in range(len(img_list)):
        img = img_list[i]
        # Recover normalization
        img = img * 255.0
        # Process img_copy and do not destroy the data of img
        img_copy = img.clone().data.permute(0, 2, 3, 1).cpu().numpy()
        img_copy = np.clip(img_copy, 0, pixel_max_cnt)
        img_copy = img_copy.astype(np.uint8)[0, :, :, :]
        img_copy = img_copy.astype(np.float32)
        img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
        # Save to certain path
        save_img_name = sample_name + '_' + name_list[i] + '.png'
        save_img_path = os.path.join(sample_folder, save_img_name)
        cv2.imwrite(save_img_path, img_copy)

def recover_process(img, height = -1, width = -1):
    img = img * 255.0
    img_copy = img.clone().data.permute(0, 2, 3, 1).cpu().numpy()
    img_copy = np.clip(img_copy, 0, 255)
    img_copy = img_copy.astype(np.uint8)[0, :, :, :]
    img_copy = img_copy.astype(np.float32)
    img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
    if (height != -1) and (width != -1):
        img_copy = cv2.resize(img_copy, (width, height))
    return img_copy

def psnr(pred, target):
    #print(pred.shape)
    #print(target.shape)
    mse = np.mean( (pred - target) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


'''
def psnr(pred, target, pixel_max_cnt = 255):
    mse = torch.mul(target - pred, target - pred)
    rmse_avg = (torch.mean(mse).item()) ** 0.5
    p = 20 * np.log10(pixel_max_cnt / rmse_avg)
    return p
'''

def grey_psnr(pred, target, pixel_max_cnt = 255):
    pred = torch.sum(pred, dim = 0)
    target = torch.sum(target, dim = 0)
    mse = torch.mul(target - pred, target - pred)
    rmse_avg = (torch.mean(mse).item()) ** 0.5
    p = 20 * np.log10(pixel_max_cnt * 3 / rmse_avg)
    return p

def ssim(pred, target):
    pred = pred.clone().data.permute(0, 2, 3, 1).cpu().numpy()
    target = target.clone().data.permute(0, 2, 3, 1).cpu().numpy()
    target = target[0]
    pred = pred[0]
    ssim = skimage.measure.compare_ssim(target, pred, multichannel = True)
    return ssim

# ----------------------------------------
#             PATH processing
# ----------------------------------------
def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

def savetxt(name, loss_log):
    np_loss_log = np.array(loss_log)
    np.savetxt(name, np_loss_log)

#rain100H/L / SPA /raindrop
def get_files(path):
    ret = []
    path_rainy = path + "/small/rain"
    path_gt = path + "/small/norain"

    for root, dirs, files in os.walk(path_rainy):
        files.sort()    
        
        for name in files:
            #if name.split('.')[1] != 'png':
            #    continue
            file_rainy = path_rainy + "/" + name
            file_gt = path_gt + "/" + name
            ret.append([file_rainy, file_gt])
    return ret
'''

#RID
def get_files(path):
    ret = []
    path_rainy = path 
    path_gt = path

    for root, dirs, files in os.walk(path_rainy):
        files.sort()
        
        for name in files:
            #if name.split('.')[1] != 'png':
            #    continue
            file_rainy = path_rainy + "/" + name
            file_gt = path_gt + "/" + name
            ret.append([file_rainy, file_gt])
    return ret
'''
'''
#rain1400
def get_files(path):
    ret = []

    path_rainy = path + "/rainy_image"
    path_gt = path + "/ground_truth"
    
    for root, dirs, files in os.walk(path_gt):
        files.sort()
        for name in files:
            if name.split('.')[1] != "jpg":
                continue
            id = name.split('.')[0]
            file_gt = path_gt + "/" + id + ".jpg"
            for i in range(1, 15):
                file_rainy = path_rainy + "/" + id + "_" + str(i) + ".jpg"
                ret.append([file_rainy, file_gt])
    return ret
'''

'''
#real
def get_files(path):
    # read a folder, return the complete path of rainy files and ground truth files
    ret=[]
    
    path_rainy = path + "/rainy_image"
    path_gt = path + "/ground_truth"

    for root, dirs, files in os.walk(path):
        files.sort()
        for name in files:
            if name.split('.')[1] != 'png':
                continue
            id = name.split('.')[0]
            file_gt = path + '/' + id + '.png'
            file_rainy = file_gt
            ret.append([file_rainy, file_gt])

    return ret
'''

'''
def get_files(path):
    # read a folder, return the complete path
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            ret.append(os.path.join(root, filespath))
    return ret
'''

def get_jpgs(path):
    # read a folder, return the image name
    ret = [] 
    for root, dirs, files in os.walk(path):
        for filespath in files:
            ret.append(filespath)
    return ret
    
def get_last_2paths(path):
    # read a folder, return the image name
    ret = [] 
    for root, dirs, files in os.walk(path):
        for filespath in files:
            if filespath[-4:] == '.png':
                wholepath = os.path.join(root, filespath)
                last_2paths = os.path.join(wholepath.split('/')[-2], wholepath.split('/')[-1])
                ret.append(last_2paths)
    return ret
    
def text_readlines(filename):
    # Try to read a txt file and return a list.Return [] if there was a mistake.
    try:
        file = open(filename, 'r')
    except IOError:
        error = []
        return error
    content = file.readlines()
    # This for loop deletes the EOF (like \n)
    for i in range(len(content)):
        content[i] = content[i][:len(content[i])-1]
    file.close()
    return content

def text_save(content, filename, mode = 'a'):
    # save a list to a txt
    # Try to save a list variable in txt file.
    file = open(filename, mode)
    for i in range(len(content)):
        file.write(str(content[i]))
    file.close()