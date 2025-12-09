#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File     :     dataSplit.py
@Time     :     2025/12/01 16:24:50
@Author   :     Louis Swift
@Desc     :     
'''

import os 
import glob
import random 


train_percent  = 0.9  # 0.9 的部分用于 训练 || 0.1 的部分用于 验证

def main():
    prefix_path = '/home/luyanlong/Desktop/CDIE/datasets/VOCdevkit/VOC2007/JPEGImages'

    filename_lst = glob.glob(prefix_path + os.sep + '*.jpg')

    num = len(filename_lst)
    list = range(num)

    train_num = int(num * train_percent)

    train_part = random.sample(list, train_num)

    f_train = open('train.txt', 'w')
    f_val = open('val.txt', 'w')

    for i in list:
        if i in train_part:
            f_train.write(filename_lst[i] + '\n')
        else:
            f_val.write(filename_lst[i] + '\n')

    f_train.close() 
    f_val.close()
    
if __name__ == '__main__':
    main()