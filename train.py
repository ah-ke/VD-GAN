"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import sys
import ntpath
from collections import OrderedDict
import os,cv2,torchvision
import torch
from options.train_options import TrainOptions
from options.test_options import TestOptions
import data
from util.iter_counter import IterationCounter
from util.visualizer import Visualizer
from util import excel
from trainers.pix2pix_trainer import Pix2PixTrainer
import numpy as np
from tqdm import tqdm

from metrics.fid.pytorch_fid.fid_score import Fid

# parse options
opt = TrainOptions().parse()

test_opt = TestOptions().parse()

# print options to help debugging
print(' '.join(sys.argv))

# load the dataset
dataloader = data.create_dataloader(opt)
test_dataloader = data.create_dataloader(test_opt)

# create trainer for our model
trainer = Pix2PixTrainer(opt)

# create tool for counting iterations
iter_counter = IterationCounter(opt, len(dataloader))

# create tool for visualization
visualizer = Visualizer(opt)

for epoch in iter_counter.training_epochs():
    iter_counter.record_epoch_start(epoch)
    for i, data_i in enumerate(dataloader, start=iter_counter.epoch_iter):
        iter_counter.record_one_iteration()

        # Training
        # train generator
        if i % opt.D_steps_per_G == 0:
            trainer.run_generator_one_step(data_i)

        # train discriminator
        trainer.run_discriminator_one_step(data_i)

        # Visualizations
        if iter_counter.needs_printing():
            losses = trainer.get_latest_losses()
            visualizer.print_current_errors(epoch, iter_counter.epoch_iter,
                                            losses, iter_counter.time_per_iter)
            visualizer.plot_current_errors(losses, iter_counter.total_steps_so_far)

        if iter_counter.needs_displaying():
            visuals = OrderedDict([('input_label', data_i['label']),
                                   ('synthesized_image', trainer.get_latest_generated()),
                                   ('real_image', data_i['image'])])
            visualizer.display_current_results(visuals, epoch, iter_counter.total_steps_so_far)

        if iter_counter.needs_saving():
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, iter_counter.total_steps_so_far))
            trainer.save('latest')
            iter_counter.record_current_iter()

    trainer.update_learning_rate(epoch)
    iter_counter.record_epoch_end()

    if epoch % opt.save_epoch_freq == 0 or \
       epoch == iter_counter.total_epochs:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, iter_counter.total_steps_so_far))
        trainer.save('latest')
        trainer.save(epoch)

    # Valing
    data_dir = os.path.join(test_opt.results_dir, opt.name)
    for i, data_i in tqdm(enumerate(test_dataloader)):
        if i * test_opt.batchSize >= test_opt.how_many:
            break

        model = trainer.pix2pix_model
        model.eval()
        generated = model(data_i, mode='inference')
        # generated = trainer.pix2pix_model(data_i, mode='inference')

        img_path = data_i['path']
        for b in range(generated.shape[0]):
            # print('process image... %s' % img_path[b])
            visuals = OrderedDict([('input_label', data_i['label'][b]),
                                   ('synthesized_image', generated[b]),
                                   ('real_image', data_i['image'][b])])
            visualizer.save_images(data_dir, visuals, img_path[b:b + 1])

    # start compute metric
    synthesized_image = os.path.join(data_dir, "synthesized_image")
    real_image = os.path.join(data_dir, "real_image")
    fid_cls = Fid([synthesized_image, real_image])
    fid_value = fid_cls()
    print("epoch:{},fid:{}\n".format(epoch,fid_value))

    # write into txt
    # with open(os.path.join(data_dir,"log.txt"),"a") as log:
    #     log.write("epoch:{},fid:{}\n".format(epoch,fid_value))

    # write into excel
    name = ["epoch", "fid"]
    value = [epoch, fid_value]
    excel_dir = os.path.join(data_dir, "log.xls")
    if epoch == 1:
        excel.write_excel_xls(excel_dir, 'sheeet1', [name])
    excel.write_excel_xls_append(excel_dir, [value])

print('Training was successfully finished.')


