"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os.path
from data.pix2pix_dataset import Pix2pixDataset
from data.image_folder import make_dataset


class CocoStuff10kDataset(Pix2pixDataset):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = Pix2pixDataset.modify_commandline_options(parser, is_train)
        parser.add_argument('--coco_no_portraits', action='store_true')
        parser.set_defaults(preprocess_mode='resize_and_crop')
        if is_train:
            size = 320 # 321
        else:
            size = 512 # 513

        parser.set_defaults(load_size=size)
        parser.set_defaults(crop_size=size)
        parser.set_defaults(display_winsize=size)
        parser.set_defaults(label_nc=182)
        parser.set_defaults(contain_dontcare_label=True)
        parser.set_defaults(cache_filelist_read=False)
        parser.set_defaults(cache_filelist_write=False)
        parser.set_defaults(no_instance=True)
        return parser

    def get_paths(self, opt):
        root = opt.dataroot
        phase = 'test' if opt.phase == 'test' else opt.phase
        # Create data list via {train, test, all}.txt
        file_list = os.path.join(root, "imageLists", phase + ".txt")
        file_list = tuple(open(file_list, "r"))

        image_paths = []
        label_paths = []
        for id_ in file_list:
            file_id = id_.rstrip()
            label_paths.append(os.path.join(root, "annotations", file_id + ".mat"))
            image_paths.append(os.path.join(root, "images", file_id + ".jpg"))

        instance_paths = []  # don't use instance map for ade20k

        return label_paths, image_paths, instance_paths
