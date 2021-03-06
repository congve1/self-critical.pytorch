from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import base64
import numpy as np
import csv
import sys
import zlib
import time
import mmap
import argparse

parser = argparse.ArgumentParser()

# output_dir
parser.add_argument('--downloaded_feats', default='data/bu_data', help='downloaded feature directory')
parser.add_argument('--output_dir', default='data/cocobu', help='output feature files')

args = parser.parse_args()

csv.field_size_limit(sys.maxsize)


FIELDNAMES = ['image_id', 'image_w','image_h','num_boxes', 'boxes', 'features']
infiles = ['test2014/test2014_resnet101_faster_rcnn_genome.tsv.0',
          'test2014/test2014_resnet101_faster_rcnn_genome.tsv.1',
          'test2014/test2014_resnet101_faster_rcnn_genome.tsv.2']

if not os.path.exists(args.output_dir+"_att"):
    os.makedirs(args.output_dir+'_att')
if not os.path.exists(args.output_dir+"_fc"):
    os.makedirs(args.output_dir+'_fc')
if not os.path.exists(args.output_dir+"_box"):
    os.makedirs(args.output_dir+'_box')

def decode_base64(data):
    """Decode base64, padding being optional.
    :param data: Base64 data as an ASCII byte string
    :returns: The decoded byte string.
    """
    missing_padding = len(data) % 4
    if missing_padding != 0:
        data += b'='* (4 - missing_padding)
    return base64.decodestring(data)

for infile in infiles:
    print('Reading ' + infile)
    with open(os.path.join(args.downloaded_feats, infile), "r+b") as tsv_in_file:
        reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames = FIELDNAMES)
        cnt = 0
        for item in reader:
            item['image_id'] = int(item['image_id'])
            item['num_boxes'] = int(item['num_boxes'])
            if len(item['features']) % 4 != 0 or len(item['boxes']) % 4 != 0:
                continue
                # print(item['image_id'])
                #pdb.set_trace()
            for field in ['boxes', 'features']:
                item[field] = np.frombuffer(decode_base64(item[field]), 
                        dtype=np.float32).reshape((item['num_boxes'],-1))
            np.savez_compressed(os.path.join(args.output_dir+'_att', str(item['image_id'])), feat=item['features'])
            np.save(os.path.join(args.output_dir+'_fc', str(item['image_id'])), item['features'].mean(0))
            np.save(os.path.join(args.output_dir+'_box', str(item['image_id'])), item['boxes'])
            cnt += 1
            if cnt % 1000 == 0 :
                print("Processed: {}".format(cnt))
