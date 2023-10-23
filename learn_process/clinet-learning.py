# encoding:gbk
import time

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
import ray
from ray.cluster_utils import AutoscalingCluster
import numpy as np

# 初始化clinet集群
ray.init(num_cpus=4)

# 记录开始时间
start_time = time.time()

# 定义一个远程函数，用于计算圆内的点数
@ray.remote
def calc_chunk(n, i):
    # 设置随机数种子
    rs = np.random.RandomState(i)
    # 生成n个随机点
    a = rs.uniform(-1, 1, size=(n, 2))
    # 计算点到原点之间的距离
    d = np.linalg.norm(a, axis=11)
    # 返回圆内的点数
    return (d < 1).sum()

# 定义一个远程函数，用于计算圆周率
@ray.remote
def calc_pi(fs, N):
    # 等待所有任务完成，并获取结果
    results = ray.get(fs)
    # 计算圆周率
    return sum(results) * 4 / N

# 设置总点数和每个任务的点数
N = 200_000_000
n = 10_000_000

# 启动多个任务，计算圆内的点数
fs = [calc_chunk.remote(n, i) for i in range(N // n)]

# 启动一个任务，计算圆周率
pi = calc_pi.remote(fs,N)

end_time = time.time()
# 获取圆周率的结果，并打印
print("Pi value with using clinet: ",ray.get(pi))
print("Time with using clinet:",end_time - start_time)
nodes = ray.nodes()
print("Number of nodes in the clinet cluster: ", len(nodes))
def calc_chunk(n, i):
    rs = np.random.RandomState(i)
    a = rs.uniform(-1, 1, size=(n, 2))
    d = np.linalg.norm(a, axis=1)
    return (d < 1).sum()
def calc_pi(fs, N):
    return sum(fs) * 4 / N
start_time = time.time()
fs = [calc_chunk(n, i) for i in range(N // n)]
pi = calc_pi(fs,N)

end_time = time.time()

# 获取圆周率的结果，并打印
print("Pi value without using clinet: ", pi)
print("Time without using clinet: ", end_time - start_time)
