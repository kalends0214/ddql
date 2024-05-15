import os
import pandas as pd
import numpy as np
import random
MEMORY_SIZE = 3000
ACTION_SPACE = 11
MAX_EPISODES = 1000

class PCMU:
    def __init__(self):
        # self.SM_dataset = self.load_dataset(dataset_path)
        self.observation_space = 3
        self.action_space = 1
        self.count_time_step = 0#更新环境影响
        self.simpling_time_step =0.1#一小时采样10个
        self.simpling_num=0#用来记录采样个数
        self.Raw_load_Yt = 700#真实电表负载需求数据
        self.Previous_Fake_load_Zt = 700#前一个真实电表负载需求数据
        self.Battery_level_Lt = 0#电池归一化容量 
        #电池电容为 10 kWh， η = 1, 充放电速率最大 4 kW的电池，容量归一化
          
    def load_dataset(self, file_name):
        # TODO: 对数据集进行预处理等操作，根据需要做相应的处理

        # 读取时间源CSV文件并提取出所有列名做时间轴
        time_path =r'C:\Users\Sliverdew\Desktop\kalends\ddql\01_summer.csv'
        time_from = pd.read_csv(time_path)
        time_row_names = time_from.columns[1:]
        time_list = time_row_names.values.tolist()
        #print(time_row_names)

        # 使用Pandas读取数据集
        df = pd.read_csv(file_name,names=['total_SM'], usecols=[0])
        # 读取每个CSV文件，修改行名称，并合并到一个DataFrame中
        #添加新列存放行名称
        df.insert(0, "time", [None] * len(df))

        #遍历行并添加行名称
        for index,row in df.iterrows():
            new_value ='{} {}'.format(file_name,time_list[index])
            df.at[index, "time"]= new_value
   
        # print(len(df))
        # 返回文件路径或其他需要返回的内容
        return df
        
    def get_data_at_current_time_step(self):
        if self.simpling_num <=len(self.SM_dataset)/360:
            current_data = self.SM_dataset['total_SM'].iloc[self.simpling_num]
            self.simpling_num += 1
            # print(self.simpling_num)
            return current_data
        else:
            return None
    
    def get_price_at_current_time(self):
        remainder = self.simpling_num % 240
        
        if (0 <= remainder <= 70) or (190 <= remainder <= 239):
            return 0.101
        elif 110 <= remainder <= 170:
            return 0.144
        elif (70 <= remainder <= 110) or (170 <= remainder <= 190):
            return 0.208
        else:
            return None  # 如果余数不在指定范围内，则返回None或其他适当的值
    
    def reset(self,file_name):
        # TODO: 重置环境状态，返回初始观察
        # 初始化用户需求负载Yt和蓄电池充电水平Lt
        self.SM_dataset = self.load_dataset(file_name)
        self.Raw_load_Yt = 700
        self.Battery_level_Lt = 0
        self.Previous_Fake_load_Zt=700
        self.simpling_num=0
        observation=np.array([self.Raw_load_Yt, self.Battery_level_Lt,self.Previous_Fake_load_Zt])
        return observation

    def step(self, action):
        print("环境第",self.count_time_step,"次更新")
        # TODO: 根据给定的动作更新环境状态，返回观察、奖励、是否终止和额外信息
        Noupdate_Fake_load=self.Previous_Fake_load_Zt
        Previous_Battery_level=self.Battery_level_Lt
        Previous_Raw_load =self.Raw_load_Yt
        # 更新用户需求负载Yt和蓄电池充电水平Lt
        #从数据中每6分钟采样一次，一天240个
        self.Raw_load_Yt = self.get_data_at_current_time_step()
        #print("当前时刻的Y:",self.Raw_load_Yt)
        #action代表充电速率∈[-4kw,4kw],电容为 10 kWh， η = 1,,理论上每6分钟最多改变4%
        Battery_C=10
        Battery_η=1
        self.Battery_level_Lt+=action*self.simpling_time_step*Battery_η/Battery_C
        # 限制电量在 [0, 1] 范围内
        if self.Battery_level_Lt < 0:
            self.Battery_level_Lt = 0
        elif self.Battery_level_Lt > 1:
            self.Battery_level_Lt = 1
        #print("当前时刻的L:",self.Battery_level_Lt)   
        
         #action为-则代表电池放电Lt减少，Yt增加；     
        Power_regulation=self.Battery_level_Lt-Previous_Battery_level
        #根据上一步产生的action对电池的作用，更新上一步的fake数据
        self.Previous_Fake_load_Zt =Previous_Raw_load-Power_regulation*Battery_C*1000/self.simpling_time_step
        #print("上一时刻的Z:",self.Previous_Fake_load_Zt) 

        # 计算奖励reward
        #①g（s,a）电池损耗:与电池充电速率成正比
        lambda_ = 0.2 # 权衡参数
        cost = self.get_price_at_current_time()  
        electricity_cost_signal=self.simpling_time_step* cost*abs(action)# 根据动作计算电力损耗
       # print("g:",electricity_cost_signal)
        #①f（s,a）fake数据与平坦化理想数值的差
        homeostasis_load=700
        privacy_leakage_signal= abs(self.Previous_Fake_load_Zt/homeostasis_load-1) # 根据动作计算隐私信息
        #print("f:",privacy_leakage_signal)
        reward = -lambda_ *electricity_cost_signal - (1-lambda_) * privacy_leakage_signal

        # 判断是否终止
        done = 0# 根据终止条件判断是否终止
        if self.Raw_load_Yt==None:
            done=1
        else:
            self.count_time_step += 1

        # 额外信息，可根据需要自定义
        info = {}

        # 更新观察
        # observation = [self.Raw_load_Yt, self.Battery_level_Lt,self.Previous_Fake_load_Zt]
        observation=np.array([self.Raw_load_Yt, self.Battery_level_Lt,self.Previous_Fake_load_Zt])
        

        return observation, reward, done, info


## 创建 PCMU 对象并进行测试

# test_path = "test_01/sm_test"
# dataset_path = test_path
# pcmu = PCMU(dataset_path)
# observation = pcmu.reset()
# done = False

# count = 0  # 计数器

# while count < 10:  # 执行10次循环
#     action = random.uniform(-4, 4)  # 生成在[-4, 4]范围内的随机动作
#     print("action:", action)
#     observation, reward, done, info = pcmu.step(action)
#     # 打印观察、奖励和是否终止等信息
#     print("Observation:", observation)
#     print("Reward:", reward)
#     print("Done:", done)
#     print("Info:", info)
#     print("***************************")
    
#     count += 1  # 计数器自增