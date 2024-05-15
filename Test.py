import tensorflow as tf
from Agent import DoubleDQN
import numpy as np
import gym
MEMORY_SIZE = 3000
ACTION_SPACE = 11

# 创建会话
sess = tf.Session()

with tf.variable_scope('Double_DQN'):
    double_DQN = DoubleDQN(
        n_actions=ACTION_SPACE, n_features=3, memory_size=MEMORY_SIZE,
        e_greedy_increment=0.001, double_q=True, sess=sess, output_graph=True
    )

# 创建Saver对象
saver = tf.train.Saver()
# 加载模型参数
model_dir = './model/my_net.ckpt'
saver.restore(sess, model_dir)

# 使用加载的模型进行预测或推理
env = gym.make("Pendulum-v0")
env = env.unwrapped
env.seed(1)
observation = env.reset()
for _ in range(1000):
    action = double_DQN.choose_action(observation)
    f_action = (action-(ACTION_SPACE-1)/2)/((ACTION_SPACE-1)/4)
    # print(action)
    observation, reward, terminated, info = env.step(np.array([f_action]))

    if terminated:
        observation = env.reset()
    # env.render()

# 关闭会话
sess.close()