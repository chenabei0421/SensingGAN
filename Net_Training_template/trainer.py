import time
import datetime
import os
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.autograd as autograd
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
#import encoding
from torchvision import transforms
import gc
import cv2
import matplotlib.pyplot as plt
import dataset
import utils
import pytorch_ssim  #SSIM Loss
import loss_functions #SA_Perceptual Loss

def Pre_train(opt):

    # ----------------------------------------
    #       Network training parameters
    # ----------------------------------------

    #torch.cuda.set_device(1)
    
    # cudnn benchmark
    cudnn.benchmark = opt.cudnn_benchmark

    # configurations
    save_folder = opt.save_path
    sample_folder = opt.sample_path
    utils.check_path(save_folder)
    utils.check_path(sample_folder)

    # Loss functions
    if opt.no_gpu == False:
        #gan_loss_func = loss_functions.GANLoss().cuda()
        criterion_L1 = torch.nn.L1Loss().cuda()
        #criterion_L2 = torch.nn.MSELoss().cuda()
        criterion_ssim = pytorch_ssim.SSIM().cuda()
        criterionSPL = loss_functions.SA_PerceptualLoss().cuda() 
    else:
        #gan_loss_func = loss_functions.GANLoss()
        criterion_L1 = torch.nn.L1Loss()
        #criterion_L2 = torch.nn.MSELoss()
        criterion_ssim = pytorch_ssim.SSIM()
        criterionSPL = loss_functions.SA_PerceptualLoss()

    # Initialize G
    generator = utils.create_generator(opt)
    #discriminator = utils.create_discriminator(opt)

    # To device
    if opt.no_gpu == False:
        if opt.multi_gpu:
            generator = nn.DataParallel(generator)
            generator = generator.cuda()
            #discriminator = nn.DataParallel(discriminator)
            #discriminator = discriminator.cuda()
        else:
            generator = generator.cuda()
            #discriminator = discriminator.cuda()

    # Optimizers
    optimizer_G = torch.optim.Adam(filter(lambda p: p.requires_grad, generator.parameters()), lr = opt.lr_g, betas = (opt.b1, opt.b2), weight_decay = opt.weight_decay)
    #optimizer_D = torch.optim.Adam(filter(lambda p: p.requires_grad, discriminator.parameters()), lr = opt.lr_d, betas = (opt.b1, opt.b2), weight_decay = opt.weight_decay)
    print("pretrained models loaded")

    # Learning rate decrease
    def adjust_learning_rate(opt, epoch, optimizer):
        target_epoch = opt.epochs - opt.lr_decrease_epoch
        remain_epoch = opt.epochs - epoch
        if epoch >= opt.lr_decrease_epoch:
            lr = opt.lr_g * remain_epoch / target_epoch
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
                
    # Learning rate decrease D
    def adjust_learning_rate_d(opt, epoch, optimizer):
        target_epoch = opt.epochs - opt.lr_decrease_epoch
        remain_epoch = opt.epochs - epoch
        if epoch >= opt.lr_decrease_epoch:
            lr = opt.lr_d * remain_epoch / target_epoch
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
    
    # Save the model if pre_train == True
    def save_model(opt, epoch, iteration, len_dataset, generator):
        """Save the model at "checkpoint_interval" and its multiple"""
        # Define the name of trained model
        if opt.save_mode == 'epoch':
            model_name = 'KPN_rainy_image_epoch%d_bs%d_generator.pth' % (epoch, opt.train_batch_size)
        if opt.save_mode == 'iter':
            model_name = 'KPN_rainy_image_iter%d_bs%d_generator.pth' % (iteration, opt.train_batch_size)
        save_model_path = os.path.join(opt.save_path, model_name)
        if opt.multi_gpu == True:
            if opt.save_mode == 'epoch':
                if (epoch % opt.save_by_epoch == 0) and (iteration % len_dataset == 0):
                    torch.save(generator.module.state_dict(), save_model_path)
                    print('The trained model generator is successfully saved at epoch %d' % (epoch))
            if opt.save_mode == 'iter':
                if iteration % opt.save_by_iter == 0:
                    torch.save(generator.module.state_dict(), save_model_path)
                    print('The trained model generator is successfully saved at iteration %d' % (iteration))
        else:
            if opt.save_mode == 'epoch':
                if (epoch % opt.save_by_epoch == 0) and (iteration % len_dataset == 0):
                    torch.save(generator.state_dict(), save_model_path)
                    print('The trained model generator is successfully saved at epoch %d' % (epoch))
            if opt.save_mode == 'iter':
                if iteration % opt.save_by_iter == 0:
                    torch.save(generator.state_dict(), save_model_path)
                    print('The trained model generator is successfully saved at iteration %d' % (iteration))
    '''               
    # Save the model if pre_train == True
    def save_model_discriminator(opt, epoch, iteration, len_dataset, discriminator):
        """Save the model at "checkpoint_interval" and its multiple"""
        # Define the name of trained model
        if opt.save_mode == 'epoch':
            model_name = 'KPN_rainy_image_epoch%d_bs%d_discriminator.pth' % (epoch, opt.train_batch_size)
        if opt.save_mode == 'iter':
            model_name = 'KPN_rainy_image_iter%d_bs%d_discriminator.pth' % (iteration, opt.train_batch_size)
        save_model_path = os.path.join(opt.save_path, model_name)
        if opt.multi_gpu == True:
            if opt.save_mode == 'epoch':
                if (epoch % opt.save_by_epoch == 0) and (iteration % len_dataset == 0):
                    torch.save(discriminator.module.state_dict(), save_model_path)
                    print('The trained model discriminator is successfully saved at epoch %d' % (epoch))
            if opt.save_mode == 'iter':
                if iteration % opt.save_by_iter == 0:
                    torch.save(discriminator.module.state_dict(), save_model_path)
                    print('The trained model discriminator is successfully saved at iteration %d' % (iteration))
        else:
            if opt.save_mode == 'epoch':
                if (epoch % opt.save_by_epoch == 0) and (iteration % len_dataset == 0):
                    torch.save(discriminator.state_dict(), save_model_path)
                    print('The trained model discriminator is successfully saved at epoch %d' % (epoch))
            if opt.save_mode == 'iter':
                if iteration % opt.save_by_iter == 0:
                    torch.save(discriminator.state_dict(), save_model_path)
                    print('The trained model discriminator is successfully saved at iteration %d' % (iteration))
'''
    # ----------------------------------------
    #             Network dataset
    # ----------------------------------------

    # Handle multiple GPUs
    #os.environ["CUDA_VISIBLE_DEVICES"] = ""    
    gpu_num = torch.cuda.device_count()
    print("There are %d GPUs used" % gpu_num)
    
    # Define the dataset
    trainset = dataset.DenoisingDataset(opt)
    print('The overall number of training images:', len(trainset))

    # Define the dataloader
    train_loader = DataLoader(trainset, batch_size = opt.train_batch_size, shuffle = True, num_workers = opt.num_workers, pin_memory = True)
    
    # ----------------------------------------
    #                 Training
    # ----------------------------------------

    # Count start time
    prev_time = time.time()
    
    # For loop training
    for epoch in range(opt.epochs):
        for i, (true_input, true_target) in enumerate(train_loader):

            #print("in epoch %d" % i)

            if opt.no_gpu == False:
                # To device
                true_input = true_input.cuda()
                true_target = true_target.cuda()
                
            true_input = Variable(true_input, requires_grad=True)
            true_target = Variable(true_target, requires_grad=True)

            # Train Generator
            fake_target, feature_map = generator(true_input, true_input)
            
            if (epoch==0 or epoch==(opt.epochs-1)) and (i%20==0):
                '''
                feature_map_hist = feature_map.view(len(true_input), 1, 1, 1, 256, 256)
                # save hist of feature_map 此batch feature map之hist (feature value數量: 256*256)
                fig=plt.figure()
                # 设置直方图的组距
                bins = np.arange(-1, 1, 0.01)
                plt.ylim(0, 10000)
                # 直方图: 各批的feature map會長一樣
                plt.hist(feature_map_hist[0].cpu().flatten().detach().numpy(), bins = bins, label = 'Feature Value', color = 'red', alpha = 0.7)
                # 设置坐标轴标签和标题
                plt.title('Batch '+ str(i) +' Red Distribution of Feature Values')
                plt.xlabel('Feature Value')
                plt.ylabel('Number of Pixels')
                
                
                # 设置坐标轴标签和标题
                plt.title('Batch '+ str(i) +' Blue Distribution of Feature Values')
                plt.xlabel('Feature Value')
                plt.ylabel('Number of Pixels')

                # 儲存圖片
                # 調整圖片之間距
                plt.subplots_adjust(left=0.125,
                    bottom=0.1, 
                    right=0.9, 
                    top=2, 
                    wspace=0.2, 
                    hspace=0.35)
                #plt.show()
                plt.tight_layout()
                print('save hist of feature map: e'+ str(epoch) +'_b' + str(i))
                plt.savefig(sample_folder+'/e'+ str(epoch) +'_b' + str(i) +'_hist.png')
                fig.clear()
                
                # plot filter channel in grayscale
                plt.imshow(feature_map_hist[0][0][0][0].cpu().detach().numpy(), cmap='gray')
                # save the feature map
                print('save the feature map e'+ str(epoch) +'_b' + str(i) + '._featuremap.png')
                plt.savefig(sample_folder+'/e'+ str(epoch) +'_b' + str(i) + '_featuremap.png')
                fig.clear()
                '''
                # save image
                utils.save_one_sample_png(sample_folder = sample_folder, sample_name = 'e'+ str(epoch) +'_b' + str(i) + '_in', img_list = true_input[0], name_list = '', pixel_max_cnt = 255)
                utils.save_one_sample_png(sample_folder = sample_folder, sample_name = 'e'+ str(epoch) +'_b' + str(i) + '_pred', img_list = fake_target[0], name_list = '', pixel_max_cnt = 255)
                utils.save_one_sample_png(sample_folder = sample_folder, sample_name = 'e'+ str(epoch) +'_b' + str(i) + '_gt', img_list = true_target[0], name_list = '', pixel_max_cnt = 255)
                
                plt.close('all')

            #GAN loss
            #O_prob = discriminator(fake_target)
            #gan_loss = gan_loss_func(O_prob, is_real=False)
            
            #Discriminator loss
            #T_prob = discriminator(true_target)
            #gt_gan_loss = gan_loss_func(T_prob, is_real=True)
            #discriminator_loss = gt_gan_loss + gan_loss

            # L1 Loss
            Pixellevel_L1_Loss = criterion_L1(fake_target, true_target)
            #Pixellevel_L2_Loss = criterion_L2(fake_target, true_target)
            
            #SSIM Loss
            ssim_loss = -criterion_ssim(true_target, fake_target)
            
            #perceptual_loss
            #perceptual_loss = criterionPL(fake_target, true_target)
            
            #SA_perceptual_loss
            SA_perceptual_loss = criterionSPL(fake_target, true_target)
            #SA_perceptual_loss = 0
            
            # Overall Loss and optimize
            generator_loss = Pixellevel_L1_Loss + 0.2*ssim_loss + 0.8*SA_perceptual_loss
            #loss = Pixellevel_L1_Loss + 0.2*ssim_loss
            #loss = Pixellevel_L1_Loss
            
            optimizer_G.zero_grad()
            generator_loss.backward()
            optimizer_G.step()
            
            #optimizer_D.zero_grad()
            #discriminator_loss.backward(retain_graph=True)
            #optimizer_D.step()

            # Determine approximate time left
            iters_done = epoch * len(train_loader) + i
            iters_left = opt.epochs * len(train_loader) - iters_done
            time_left = datetime.timedelta(seconds = iters_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            print("\r[Epoch %d/%d] [Batch %d/%d] [Loss: %.4f %.4f %.4f] Time_left: %s" %
                (epoch, opt.epochs, i, len(train_loader), Pixellevel_L1_Loss.item(), ssim_loss.item(), SA_perceptual_loss, time_left))

            # Save model at certain epochs or iterations
            save_model(opt, (epoch+1), (iters_done + 1), len(train_loader), generator)
            #save_model_discriminator(opt, (epoch+1), (iters_done + 1), len(train_loader), discriminator)

            # Learning rate decrease at certain epochs
            adjust_learning_rate(opt, epoch, optimizer_G)
            #adjust_learning_rate_d(opt, epoch, optimizer_D)

            gc.collect()