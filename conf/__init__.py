# encoding:gbk
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
# 动态加载设置
import conf.global_settings as settings

class Settings:
    def __init__(self, settings):

        for attr in dir(settings):
            if attr.isupper():
                setattr(self, attr, getattr(settings, attr))

settings = Settings(settings)