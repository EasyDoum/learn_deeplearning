'''
Author: douge 1041790491@qq.com
Date: 2023-08-10 11:53:07
LastEditors: douge 1041790491@qq.com
LastEditTime: 2023-08-14 18:29:24
FilePath: /douge/learn_deeplearning/python神经网络编程学习/neural_network.py
Description: 定义一个3层神经网络，可以控制输入层、隐藏层和输出层的节点数量，并实现
输出计算和反向传播以达到训练的目的

Copyright (c) 2023 by douge, All Rights Reserved. 
'''

import numpy as np
import scipy.special

# 定义一个神经网络类
class NeuralNetwork:

    def __init__(self, inodes, hnodes, onodes, lr) -> None:
        '''
        description: 初始化函数，初始化神经网络的形状：输入节点、隐藏节点、输出节点、学习率
        param self: 
        param inodes: 输入层节点数
        param hnodes: 隐藏层节点数
        param onodes: 输出层节点数
        param lr: 学习率
        return: 无返回值 
        '''
        
        self.inodes = inodes
        self.hnodes = hnodes
        self.onodes = onodes

        self.lr = lr

        # 输入-->隐藏层权重矩阵，随机正态分布
        self.w_ih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        # 隐藏-->输出层权重矩阵，随机正态分布
        self.w_ho = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

        # 激活函数
        self.active_func = lambda x: scipy.special.expit(x)
        pass


    def learn(self):
        pass


    def query(self, input_list):

        input = np.array(input_list)
        pass

