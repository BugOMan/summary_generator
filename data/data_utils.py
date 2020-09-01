#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pathlib
import sys

abs_path = pathlib.Path(__file__).parent.absolute()
sys.path.append(sys.path.append(abs_path))


def read_samples(filename):
    """Read the data file and return a sample list.

    Args:
        filename (str): The path of the txt file.

    Returns:
        list: A list conatining all the samples in the file.
    """
    samples = []
    with open(filename, 'r', encoding='utf8') as file:
        for line in file:
            samples.append(line.strip())
    return samples


def write_samples(samples, file_path, opt='w'):
    """Write the samples into a file.

    Args:
        samples (list): The list of samples to write.
        file_path (str): The path of file to write.
        opt (str, optional): The "mode" parameter in open(). Defaults to 'w'.
    """
    with open(file_path, opt, encoding='utf8') as file:
        for line in samples:
            file.write(line)
            file.write('\n')


def partition(samples):
    """Partition a whole sample set into training set, dev set and test set.

    Args:
        samples (Iterable): The iterable that holds the whole sample set.
    """
    train, dev, test = [], [], []
    count = 0
    for sample in samples:
        count += 1
        if count % 1000 == 0:
            print(count)
        if count <= 1000:  # Test set size.
            test.append(sample)
        elif count <= 6000:  # Dev set size.
            dev.append(sample)
        else:
            train.append(sample)
    print('train: ', len(train))

    write_samples(train, os.path.join(abs_path, '../files/train.txt'))
    write_samples(dev, os.path.join(abs_path, '../files/dev.txt'))
    write_samples(test, os.path.join(abs_path, '../files/test.txt'))


def isChinese(word):
    """Distinguish Chinese words from non-Chinese ones.

    Args:
        word (str): The word to be distinguished.

    Returns:
        bool: Whether the word is a Chinese word.
    """
    for ch in word:
        if '\u4e00' <= ch <= '\u9fff':
            return True
    return False
