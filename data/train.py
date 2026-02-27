# import time
# import torch
# from options.train_options import TrainOptions
# from data import create_dataset
# from models import create_model
# from util.visualizer import Visualizer
#
#
# if __name__ == '__main__':
#     start = time.time()
#
#     opt = TrainOptions().parse()   # get training options
#     dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
#     opt.dataset_size = len(dataset)    # get the number of images in the dataset.
#
#     model = create_model(opt)      # create a model given opt.model and other options
#     print('The number of training images = %d' % opt.dataset_size)
#
#     visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
#     opt.visualizer = visualizer
#     total_iters = 0                # the total number of training iterations
#
#     optimize_time = 0.1
#     times = []
#     for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
#         epoch_start_time = time.time()  # timer for entire epoch
#         iter_data_time = time.time()    # timer for data loading per iteration
#         epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
#         visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
#
#         dataset.set_epoch(epoch)
#         model.set_epoch(epoch)
#         for i, data in enumerate(dataset):  # inner loop within one epoch
#             iter_start_time = time.time()  # timer for computation per iteration
#             if total_iters % opt.print_freq == 0:
#                 t_data = iter_start_time - iter_data_time
#
#             batch_size = data["A"].size(0)
#             total_iters += batch_size
#             epoch_iter += batch_size
#             if len(opt.gpu_ids) > 0:
#                 torch.cuda.synchronize()
#             optimize_start_time = time.time()
#             if epoch == opt.epoch_count and i == 0:
#                 model.data_dependent_initialize(data)
#                 model.setup(opt)               # regular setup: load and print networks; create schedulers
#                 model.parallelize()
#                 # model.print_networks(True)
#             model.set_input(data)  # unpack data from dataset and apply preprocessing
#             model.optimize_parameters()   # calculate loss functions, get gradients, update network weights
#             if len(opt.gpu_ids) > 0:
#                 torch.cuda.synchronize()
#             optimize_time = (time.time() - optimize_start_time) / batch_size * 0.005 + 0.995 * optimize_time
#
#             if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
#                 save_result = total_iters % opt.update_html_freq == 0
#                 model.compute_visuals()
#                 visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)
#
#             if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
#                 losses = model.get_current_losses()
#                 visualizer.print_current_losses(epoch, epoch_iter, losses, optimize_time, t_data)
#                 if opt.display_id is None or opt.display_id > 0:
#                     visualizer.plot_current_losses(epoch, float(epoch_iter) / opt.dataset_size, losses)
#
#             if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
#                 print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
#                 print(opt.name)  # it's useful to occasionally show the experiment name on console
#                 save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
#                 model.save_networks(save_suffix)
#
#             iter_data_time = time.time()
#
#         if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
#             print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
#             model.save_networks('latest')
#             model.save_networks(epoch)
#
#         print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
#         model.update_learning_rate()                     # update learning rates at the end of every epoch.
#
#     m, s = divmod(time.time() - start, 60)
#     h, m = divmod(m, 60)
#     print(f"{opt.name} >>>> Training completed in {h:.1f}hrs, {m:.1f}min, {s:.1f}sec!")
#     model.save_texts(f"Training completed in {h:.1f}hrs, {m:.1f}min, {s:.1f}sec!", 'training_time')

import time
import torch
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
import os
import numpy as np
from  skimage.metrics  import  peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def nmse(imageA, imageB):
    err1 = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err1 /= np.sum((imageA.astype("float")) ** 2)
    return err1
def print_log(logger,message):
    print(message, flush=True)
    if logger:
        logger.write(str(message) + '\n')
if __name__ == '__main__':
    start = time.time()

    opt = TrainOptions().parse()  # get training options
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    opt.dataset_size = len(dataset)  # get the number of images in the dataset.

    ##logger ##
    save_dir = os.path.join(opt.checkpoints_dir, opt.name)
    logger = open(os.path.join(save_dir, 'log.txt'), 'w+')
    print_log(logger,opt.name)
    logger.close()
    #validation data
    opt.phase='val'
    data_loader_val = create_dataset(opt)
    dataset_val = data_loader_val.load_data()
    dataset_size_val = len(data_loader_val)
    print('#Validation images = %d' % dataset_size_val)
    if opt.model=='QAmo':
        L1_avg=np.zeros([2,opt.n_epochs + opt.n_epochs_decay,len(dataset_val)])
        psnr_avg=np.zeros([2,opt.n_epochs + opt.n_epochs_decay,len(dataset_val)])
        ssim_avg = np.zeros([2,opt.n_epochs + opt.n_epochs_decay, len(dataset_val)])
        nmse_avg = np.zeros([2,opt.n_epochs + opt.n_epochs_decay, len(dataset_val)])
    else:
        L1_avg=np.zeros([opt.n_epochs + opt.n_epochs_decay,len(dataset_val)])
        psnr_avg=np.zeros([opt.n_epochs+ opt.n_epochs_decay,len(dataset_val)])
        ssim_avg=np.zeros([opt.n_epochs+ opt.n_epochs_decay,len(dataset_val)])
        nmse_avg=np.zeros([opt.n_epochs+ opt.n_epochs_decay,len(dataset_val)])

    model = create_model(opt)  # create a model given opt.model and other options
    print('The number of training images = %d' % opt.dataset_size)
    visualizer = Visualizer(opt)  # create a visualizer that display/save images and plots
    opt.visualizer = visualizer
    total_iters = 0  # the total number of training iterations

    optimize_time = 0.1
    times = []
    for epoch in range(opt.epoch_count,
                       opt.n_epochs + opt.n_epochs_decay + 1):  # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()  # timer for data loading per iteration
        epoch_iter = 0  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()  # reset the visualizer: make sure it saves the results to HTML at least once every epoch

        dataset.set_epoch(epoch)
        model.set_epoch(epoch)
        opt.phase = 'train'
        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            batch_size = data["A"].size(0)
            total_iters += batch_size
            epoch_iter += batch_size
            if len(opt.gpu_ids) > 0:
                torch.cuda.synchronize()
            optimize_start_time = time.time()
            if epoch == opt.epoch_count and i == 0:
                model.data_dependent_initialize(data)
                model.setup(opt)  # regular setup: load and print networks; create schedulers
                model.parallelize()
            model.set_input(data)  # unpack data from dataset and apply preprocessing
            model.optimize_parameters()  # calculate loss functions, get gradients, update network weights
            if len(opt.gpu_ids) > 0:
                torch.cuda.synchronize()
            optimize_time = (time.time() - optimize_start_time) / batch_size * 0.005 + 0.995 * optimize_time

            if total_iters % opt.display_freq == 0:  # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:  # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                visualizer.print_current_losses(epoch, epoch_iter, losses, optimize_time, t_data)
                if opt.display_id is None or opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / opt.dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:  # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                print(opt.name)  # it's useful to occasionally show the experiment name on console
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()
            ######
            ####Validaition step
        if epoch % opt.save_epoch_freq == 0:
            logger = open(os.path.join(save_dir, 'log.txt'), 'a')
            print(opt.dataset_mode)
            opt.phase = 'val'
            for i, data_val in enumerate(dataset_val):
                #
                model.set_input(data_val)
                #
                model.test()
                #
                fake_im_B = model.fake_B.cpu().data.numpy()
                #
                real_im_B = model.real_B.cpu().data.numpy()

                # fake_im_A = model.fake_A.cpu().data.numpy()
                # #
                # real_im_A = model.real_A.cpu().data.numpy()

                real_im_B = real_im_B * 0.5 + 0.5
                fake_im_B = fake_im_B * 0.5 + 0.5

                # real_im_A = real_im_A * 0.5 + 0.5
                # fake_im_A = fake_im_A * 0.5 + 0.5

                if real_im_B.max() <= 0:
                    continue
                L1_avg[0,epoch - 1, i] = abs(fake_im_B - real_im_B).mean()
                psnr_avg[0,epoch - 1, i] = psnr(fake_im_B / fake_im_B.max(), real_im_B / real_im_B.max())
                # psnr_avg[epoch - 1, i] = psnr(fake_im, real_im)
                ssim_avg[0,epoch - 1, i] = ssim(fake_im_B.squeeze().squeeze(), real_im_B.squeeze().squeeze(),data_range=1)
                nmse_avg[0,epoch-1, i] =nmse(fake_im_B, real_im_B)

                # L1_avg[1, epoch - 1, i] = abs(fake_im_A - real_im_A).mean()
                # psnr_avg[1, epoch - 1, i] = psnr(fake_im_A / fake_im_A.max(), real_im_A / real_im_A.max())
                # # psnr_avg[epoch - 1, i] = psnr(fake_im, real_im)
                # ssim_avg[1, epoch - 1, i] = ssim(fake_im_A.squeeze().squeeze(), real_im_A.squeeze().squeeze(),
                #                                  data_range=1)
                # nmse_avg[1, epoch - 1, i] = nmse(fake_im_A, real_im_A)

            #
            l1_avg_loss_B = np.mean(L1_avg[0,epoch - 1])
            mean_psnr_B = np.mean(psnr_avg[0,epoch - 1])
            std_psnr_B = np.std(psnr_avg[0,epoch - 1])
            mean_ssim_B = np.mean(ssim_avg[0,epoch - 1])
            mean_nmse_B = np.mean(nmse_avg[0, epoch - 1])

            # l1_avg_loss_A = np.mean(L1_avg[1, epoch - 1])
            # mean_psnr_A = np.mean(psnr_avg[1, epoch - 1])
            # std_psnr_A = np.std(psnr_avg[1, epoch - 1])
            # mean_ssim_A = np.mean(ssim_avg[1, epoch - 1])
            # mean_nmse_A = np.mean(nmse_avg[1, epoch -1])
            print_log(logger, 'Epoch %3d   l1_avg_loss_B: %.5f   mean_psnr_B: %.3f  std_psnr_B:%.3f  mean_ssim_B:%.3f mean_nmse_B: %.3f '  % \
                      (epoch, l1_avg_loss_B, mean_psnr_B, std_psnr_B, mean_ssim_B, mean_nmse_B))

            # print_log(logger,
            #           'Epoch %3d   l1_avg_loss_A: %.5f   mean_psnr_A: %.3f  std_psnr_A:%.3f  mean_ssim_A:%.3f mean_nmse_A: %.3f ' % \
            #           (epoch, l1_avg_loss_A, mean_psnr_A, std_psnr_A, mean_ssim_A, mean_nmse_A))
            #
            print_log(logger, '')
            logger.close()
        #######
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        # if epoch % opt.save_epoch_freq == 0:  # cache our model every <save_epoch_freq> epochs
        #     print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
        #     model.save_networks('latest')
        #     model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (
        epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
        model.update_learning_rate()  # update learning rates at the end of every epoch.

    m, s = divmod(time.time() - start, 60)
    h, m = divmod(m, 60)
    print(f"{opt.name} >>>> Training completed in {h:.1f}hrs, {m:.1f}min, {s:.1f}sec!")
    model.save_texts(f"Training completed in {h:.1f}hrs, {m:.1f}min, {s:.1f}sec!", 'training_time')
