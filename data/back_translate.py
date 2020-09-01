#!/usr/bin/python
# -*- coding: UTF-8 -*-
# !pip3 install jieba==0.36.2
import jieba
import http.client
import hashlib
import urllib
import random
import json
import time
from data_utils import write_samples

import os


def translate(q, source, target):
    """translate q from source language to target language

    Args:
        q (str): sentence
        source(str): The language code
        target(str): The language code
    Returns:
        (str): result of translation
    """
    # Please refer to the official documentation   https://api.fanyi.baidu.com/  
    # There are demo on the website ,  register on the web site ,and get AppID, key, python3 demo.
    appid = '20200819000547028'  # Fill in your AppID
    secretKey = 'Br31bKUxSsNrzYmSbcg2'  # Fill in your key

    ###########################################
    #          TODO: module 2 task 1          #
    ###########################################
    httpClient = None
    myurl = '/api/trans/vip/translate'
    fromLang = source  # The original language
    toLang = target  # 译文语种
    salt = random.randint(32768, 65536)
    # q = 'apple'
    sign = appid + q + str(salt) + secretKey
    sign = hashlib.md5(sign.encode()).hexdigest()
    myurl = myurl + '?appid=' + appid + '&q=' + urllib.parse.quote(q) + \
            '&from=' + fromLang + '&to=' + toLang + \
            '&salt=' + str(salt) + '&sign=' + sign
    try:
        httpClient = http.client.HTTPConnection('api.fanyi.baidu.com')
        httpClient.request('GET', myurl)

        # response是HTTPResponse对象
        response = httpClient.getresponse()
        result_all = response.read().decode("utf-8")
        result = json.loads(result_all)

        print(result)
        return result
    except Exception as e:
        print(e)
    finally:
        if httpClient:
            httpClient.close()


def back_translate(q):
    """back_translate

    Args:
        q (str): sentence

    Returns:
        (str): result of back translation
    """
    ###########################################
    #          TODO: module 2 task 2          #
    ###########################################
    target = translate(q, 'zh', 'en')
    dst = target['trans_result'][0]['dst']
    time.sleep(1)
    dst = translate(dst, 'en', 'zh')['trans_result'][0]['dst']
    time.sleep(1)
    return dst

def translate_continue(sample_path, translate_path):
    """translate  original file to new file

    Args:
        sample_path (str): original file path
        translate_path (str): target file path
    Returns:
        (str): result of back translation
    """
    ###########################################
    #          TODO: module 2 task 3          #
    ###########################################
    translate_list = []

    with open(sample_path, encoding='utf-8') as fr:
        lines = fr.readlines()
        # refs = [(sample.split('<sep>')[1].split() for sample in lines]
        for item in lines:
            items = item.split('<sep>')
            translate_str = back_translate(items[1])
            translate_list.append(items[0] + '<sep>' + translate_str + '\n')

    with open(translate_path, 'w', encoding='utf-8') as fw:
        fw.writelines(translate_list)

if __name__ == '__main__':
    sample_path = 'output/train.txt'
    translate_path = 'output/translated.txt'
    translate_continue(sample_path, translate_path)