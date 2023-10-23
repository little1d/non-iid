## 目录结构：
```README
D:.
│  dataset.py
│  Ir_finder.py
│  list.txt
│  non-iid-data.py
│  README.md
│  test.py
│  train.py
│  utils.py
│  
├─checkpoint
│  └─resnet50
│      ├─Saturday_21_October_2023_10h_45m_50s
│      │      resnet50-10-regular.pth
│      │      resnet50-20-regular.pth
│      │      
│      └─Saturday_21_October_2023_16h_57m_12s
├─CIFAR100
│  │  cifar-10-python.tar.gz
│  │  
│  └─cifar-10-batches-py
│          batches.meta
│          data_batch_1
│          data_batch_2
│          data_batch_3
│          data_batch_4
│          data_batch_5
│          readme.html
│          test_batch
│          
├─conf
│  │  global_settings.py
│  │  __init__.py
│  │  
│  └─__pycache__
│          global_settings.cpython-311.pyc
│          __init__.cpython-311.pyc
├─learn_process
│      clinet-learning.py
│      
├─runs
│  └─resnet50
│      ├─Saturday_21_October_2023_10h_43m_11s
│      │      events.out.tfevents.1697856218.Harrison.26984.0
│      │      
│      ├─Saturday_21_October_2023_10h_45m_50s
│      │      events.out.tfevents.1697856351.Harrison.25756.0
│      │      
│      └─Saturday_21_October_2023_16h_57m_12s
│              events.out.tfevents.1697878634.Harrison.14240.0 
```
 

# 运行训练文件 `train.py`
```
# 不使用gpu 使用cpu
python train.py -net resnet50

#使用gpu
python train.py -net resnet50 -gpu
```

```
#启动tensorboard
pip install tensorborad
tensorboard --logdir=./
```
# 运行测试文件 test.py
```
 python test.py -net resnet50 -weights .\checkpoint\resnet50\Saturday_21_October_2023_10h_45m_50s\resnet50-10-regular.pth 
```



***
# `lr_finder.py`

***用于寻找最佳学习率的python脚本，用于训练神经网络，使用一种称为学习率调度的技术，该技术可以在训练过程中动态地调整学习率***

详细解释：
1. 导入必要的库
2. 定义FindLR类：这个类是一个自定义的学习率调度器，它会在每个迭代中逐渐增加学习率。这对于找到最佳学习率非常有用
3. 解析命令行参数：这部分代码接收用户输入的参数，如网络类型、批处理大小、最小和最大学习率、迭代次数等。
4. 数据加载：使用预定义的函数从CIFAR100数据集中加载训练数据
5. 网络和优化器设置：根据用户输入的参数创建网络，并设置SGD优化器和损失函数。
6. 训练过程：在每个批次上运行网络，计算损失，并使用优化器更新网络权重。同时，更新学习率并记录下来
7. 结果可视化：最后，将损失和学习率的关系绘制成图形，并保存为jpg文件。

# `train.py`

***用于训练和评估神经网络模型给的脚本，使用PyTorch库来构建和训练模型，并使用TensorBoard来可视化训练过程和结果***

详细解释：
1. 导入所需的库和模块，还导入一些自定义的辅助函数和设置
2.定义一个train函数，用于执行模型的训练过程。在训练过程中，首先将模型设置为训练模式，然后遍历训练数据集的批次。对于每个批次，执行向前传播、计算损失、反向传播和参数更新的步骤。在训练过程中，还记录了一些训练指标，如损失和梯度，并将他们写入Tensorboard中。
3. 定义一个`eval_training`函数，用于评估模型在测试数据集上的性能，在评估过程中，将模型设置为评估模式，并遍历测试数据集的批次，对于每个批次，执行前向传播、计算损失和准确率的步骤，并累计总的损失和正确预测的数量。最后，计算平均损失和准确率，并打印出来
4. 定义一些命令行参数，如批次大小、学习率、是否恢复训练等。然后使用这些参数来创建模型、加载数据集、定义损失函数和优化器，并设置学习率调度器和权重衰减。如果指定了恢复训练的选项，它会加载最近的模型权重和训练状态。
5. 进入主训练循环，遍历指定的训练轮数，在每个训练轮中，执行训练和聘雇过程，根据指定条件保存最佳性能的模型权重
6. 最后，代码使用`torch.save`函数将模型的状态字典保存到指定的权重文件中，并关闭TensorBoard的写入器

# `test.py`

***一个用于测试神经网络性能的脚本***
1. 导入必要的库和模块
2. 解析命令行参数
3. 加载网络模型和测试数据集
4. 加载权重文件和设置网络为评估模式，使用`torch.load`函数加载指定的权重文件，并使用`lad_state_dict`方法将权重加载到网络模型中，调用`eval`方法将网络设置为评估模式，即网络不会进行梯度计算和参数更新
5. 进行计算前5个类别最高概率的类别，并将真实标签进行扩展，打印准确率

# `utils.py`
存储一些公用函数

# `checkpoint`
每轮训练时的权重与正确率数据

# `learn_process.py`
自己学习的时候写的一点点代码，帮助我理解相关模型与算法

# `runs.py`
日志文件，可在tensorboard上可视化