# encoding:gbk
import time

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
import ray
from ray.cluster_utils import AutoscalingCluster
import numpy as np

# ��ʼ��clinet��Ⱥ
ray.init(num_cpus=4)

# ��¼��ʼʱ��
start_time = time.time()

# ����һ��Զ�̺��������ڼ���Բ�ڵĵ���
@ray.remote
def calc_chunk(n, i):
    # �������������
    rs = np.random.RandomState(i)
    # ����n�������
    a = rs.uniform(-1, 1, size=(n, 2))
    # ����㵽ԭ��֮��ľ���
    d = np.linalg.norm(a, axis=11)
    # ����Բ�ڵĵ���
    return (d < 1).sum()

# ����һ��Զ�̺��������ڼ���Բ����
@ray.remote
def calc_pi(fs, N):
    # �ȴ�����������ɣ�����ȡ���
    results = ray.get(fs)
    # ����Բ����
    return sum(results) * 4 / N

# �����ܵ�����ÿ������ĵ���
N = 200_000_000
n = 10_000_000

# ����������񣬼���Բ�ڵĵ���
fs = [calc_chunk.remote(n, i) for i in range(N // n)]

# ����һ�����񣬼���Բ����
pi = calc_pi.remote(fs,N)

end_time = time.time()
# ��ȡԲ���ʵĽ��������ӡ
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

# ��ȡԲ���ʵĽ��������ӡ
print("Pi value without using clinet: ", pi)
print("Time without using clinet: ", end_time - start_time)
