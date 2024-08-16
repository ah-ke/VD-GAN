"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os
from collections import OrderedDict

import data
from options.test_options import TestOptions
from models.pix2pix_model import Pix2PixModel
from util.visualizer import Visualizer
from util import html

opt = TestOptions().parse()

dataloader = data.create_dataloader(opt)

model = Pix2PixModel(opt)
model.eval()
model = model.cuda()

visualizer = Visualizer(opt)


# test
# data_dir = os.path.join(opt.results_dir, opt.name)
data_dir = r"G:\TransGAN\generator\picture\kontrol\result"
for i, data_i in enumerate(dataloader):
    if i * opt.batchSize >= opt.how_many:
        break
    # if "ADE_val_00000677" not in data_i['path'][0]:
    #     continue
    generated = model(data_i, mode='inference')
    img_path = data_i['path']

    for b in range(generated.shape[0]):
        print('process image... %s' % img_path[b])
        visuals = OrderedDict([('input_label', data_i['label'][b]),
                               ('synthesized_image', generated[b]),
                               ('real_image', data_i['image'][b])])
        visualizer.save_images(data_dir, visuals, img_path[b:b + 1])

# start compute metric
synthesized_image = os.path.join(data_dir, "synthesized_image")
real_image = os.path.join(data_dir, "real_image")
# from metrics.fid.pytorch_fid.fid_score import Fid
# fid_cls = Fid([synthesized_image, real_image])
# fid_value = fid_cls()
# fid_value = round(fid_value, 2)
# print(fid_value)

import metrics
segment = metrics.create_metricloader(opt, data_dir)
map = segment()

name = ["mIoU", "mAccu", "fid"]
value = [map["mIoU"], map["mAccu"], fid_value]
metric_dict = dict(zip(name, value))
print(metric_dict)



