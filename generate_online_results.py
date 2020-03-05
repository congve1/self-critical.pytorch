from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import numpy as np

import time
import copy
import os
from six.moves import cPickle

import opts
import models
from dataloader import *
from dataloaderraw import *
import eval_utils
import argparse
import misc.utils as utils
import torch

# Input arguments and options
parser = argparse.ArgumentParser()
# Input paths
parser.add_argument('--model', type=str, default='',
                help='path to model to evaluate')
parser.add_argument('--cnn_model', type=str,  default='resnet101',
                help='resnet101, resnet152')
parser.add_argument('--alg_name', type=str, default='clw',
                    help='alg_name')
parser.add_argument('--infos_path', type=str, default='',
                help='path to infos to evaluate')
opts.add_eval_options(parser)

opt = parser.parse_args()

# Load infos
with open(opt.infos_path) as f:
    infos = utils.pickle_load(f)

# override and collect parameters
replace = ['input_fc_dir', 'input_att_dir', 'input_box_dir', 'input_label_h5', 'input_json', 'batch_size', 'id']
ignore = ['start_from']

for k in vars(infos['opt']).keys():
    if k in replace:
        setattr(opt, k, getattr(opt, k) or getattr(infos['opt'], k, ''))
    elif k not in ignore:
        if not k in vars(opt):
            vars(opt).update({k: vars(infos['opt'])[k]}) # copy over options from model

vocab = infos['vocab'] # ix -> word mapping

# Setup the model
opt.vocab = vocab
model = models.setup(opt)
del opt.vocab
model.load_state_dict(torch.load(opt.model))
model.cuda()
model.eval()
crit = utils.LanguageModelCriterion()

# Create the Data Loader instance
val_opt = copy.deepcopy(opt)
val_opt.input_fc_dir = "data/cocobu_fc"
val_opt.input_att_dir = "data/cocobu_att"
val_opt.input_json = "data/captions_val2014.json"
val_opt.input_label_h5 = 'none'
val_opt.split = "val"
val_dataloader = DataLoader(val_opt)

test_opt = copy.deepcopy(opt)
test_opt.input_fc_dir = "data/cocobu_fc"
test_opt.input_att_dir = "data/cocobu_att"
test_opt.input_json = "data/captions_test2014.json"
test_opt.input_label_h5 = 'none'
test_opt.split = "test"

test_dataloader = DataLoader(test_opt)
# When eval using provided pretrained model, the vocab may be different from what you have in your cocotalk.json
# So make sure to use the vocab in infos file.


# Set sample options
for loader, opt, name in zip([val_dataloader, test_dataloader], [val_opt, test_opt], [opt.agl_name, opt.alg_name]):
    loader.ix_to_word = infos['vocab']
    opt.datset = opt.input_json
    loss, split_predictions, lang_stats = eval_utils.eval_split(model, crit, loader,
        vars(opt))
    file_path = "captions_"+name+"_clw_results.json"
    print("writing results to {}".format(file_path))
    json.dump(split_predictions, open(file_path, "w"))

