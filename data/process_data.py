#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import pathlib
import jieba
import pandas as pd

abs_path = pathlib.Path(__file__).parent.absolute()
sys.path.append(sys.path.append(abs_path))
from data_utils import write_samples, partition


samples = set()
# Read csv file.
origin_list = ['../files/train.csv', '../files/eval.csv', '../files/test.csv']
pair_list = ['../files/train.txt', '../files/eval.txt', '../files/test.txt']

for i in range(len(origin_list)):
    origin_path = os.path.join(abs_path, origin_list[i])
    origin_data = pd.read_csv(origin_path, header=None)
    origin_data.columns = ['title', 'descri']
    print(origin_data.head())
    for index, row in origin_data.iterrows():
        title = ' '.join(list(jieba.cut(row[0])))
        descri = ' '.join(list(jieba.cut(row[1])))
        sample = title + '<sep>' + descri
        samples.add(sample)
    write_path = os.path.join(abs_path, pair_list[i])
    write_samples(samples, write_path)