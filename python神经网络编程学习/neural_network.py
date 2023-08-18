"""
Author: douge 1041790491@qq.com
Date: 2023-08-10 11:53:07
LastEditors: douge 1041790491@qq.com
LastEditTime: 2023-08-17 10:41:51
FilePath: /douge/learn_deeplearning/python神经网络编程学习/neural_network.py
Description: 定义一个3层神经网络，可以控制输入层、隐藏层和输出层的节点数量，并实现
输出计算和反向传播以达到训练的目的

Copyright (c) 2023 by douge, All Rights Reserved. 
"""

import numpy as np
import scipy.special


# 定义一个神经网络类
class NeuralNetwork:
    def __init__(self, inodes, hnodes, onodes, lr) -> None:
        """
        description: 初始化函数，初始化神经网络的形状：输入节点、隐藏节点、输出节点、学习率
        param self:
        param inodes: 输入层节点数
        param hnodes: 隐藏层节点数
        param onodes: 输出层节点数
        param lr: 学习率
        return: 无返回值
        """

        self.inodes = inodes
        self.hnodes = hnodes
        self.onodes = onodes

        self.lr = lr

        # 输入-->隐藏层权重矩阵，随机正态分布，均值为0，方差为隐藏层节点数量，形状为H*I
        self.w_ih = np.random.normal(
            0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes)
        )
        # 隐藏-->输出层权重矩阵，随机正态分布，均值为0，方差为输出层节点数量，形状为O*H
        self.w_ho = np.random.normal(
            0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes)
        )

        # 激活函数
        self.active_func = lambda x: scipy.special.expit(x)

    def learn(self, input_list, target_list):
        """
        description: 学习函数，实现3层神经网络的输出计算与反向传播
        param self:
        param input_list: 输入数据集
        param target_list: 目标值数据集
        return: 无返回值
        """

        # 输入、目标值列表转换为ndarray
        input = np.array(input_list, ndmin=2).T
        target = np.array(target_list, ndmin=2).T

        # 获取隐藏层输出
        h_output = self.get_h_output(input)

        # 获取输出层输出
        output = self.get_output(h_output)

        # 获取输出误差
        o_error = target - output

        # 获取隐藏层输出
        h_error = np.dot(self.w_ho.T, o_error)

        # 修改隐藏-->输出层权重矩阵
        self.w_ho += self.lr * np.dot(
            (o_error * output * (1 - output)), np.transpose(h_output)
        )

        # 修改输入-->隐藏层权重矩阵
        self.w_ih += self.lr * np.dot(
            h_error * h_output * (1 - h_output), np.transpose(input)
        )

    def query(self, input_list):
        """
        description: 查询函数，打印输入节点数据、隐藏层输出、输出节点数据
        param self
        param input_list: 输入的数据列表
        return: 无返回值
        """

        input = np.array(input_list, ndmin=2).T
        print(f"输入节点数据：\n{input}")

        h_output = self.get_h_output(input)
        print(f"隐藏层输出：\n{h_output}")

        output = self.get_output(h_output)
        print(f"输出节点数据：\n{output}")

    def get_output(self, h_output):
        """
        description: 输入隐藏层输出，计算并输出最终输出层
        param self
        param h_output: 隐藏层输出
        return: 返回输出层
        """

        # 权重矩阵和隐藏层的点积
        o_input = np.dot(self.w_ho, h_output)

        # 应用激活函数
        output = self.active_func(o_input)

        return output

    def get_h_output(self, input):
        """
        description: 传入输入层数据，计算并输出隐藏层
        param self
        param input: 输入层数据
        return: 返回隐藏层输出
        """

        # 权重矩阵和输入层的点积
        h_input = np.dot(self.w_ih, input)

        # 应用激活函数
        h_output = self.active_func(h_input)

        return h_output


if __name__ == "__main__":
    i_nodes = 6
    h_nodes = 4
    o_nodes = 3

    lr = 0.3
    n = NeuralNetwork(i_nodes, h_nodes, o_nodes, lr)
    data = [1.5, 3.6, 4.8, -1.6, 20, -5.9]
    target_data = [0.7, 0.4, 0.9]
    # res1 = n.query(data)
    i = 1000
    while i > 0:
        res2 = n.learn(data, target_data)
        i -= 1
    res3 = n.query(data)

    # print(type(res))
    # print(res)
