import gym
from Agent import DoubleDQN
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from env import PCMU
import os


# env = gym.make('Pendulum-v0')


env = PCMU()
#env = env.unwrapped
#env.seed(1)
MEMORY_SIZE = 10000
ACTION_SPACE = 11

sess = tf.Session()
with tf.variable_scope('Natural_DQN'):
    natural_DQN = DoubleDQN(
        n_actions=ACTION_SPACE, n_features=3, memory_size=MEMORY_SIZE,
        e_greedy_increment=0.001, double_q=False, sess=sess
    )

with tf.variable_scope('Double_DQN'):
    double_DQN = DoubleDQN(
        n_actions=ACTION_SPACE, n_features=3, memory_size=MEMORY_SIZE,
        e_greedy_increment=0.001, double_q=True, sess=sess, output_graph=True)

sess.run(tf.global_variables_initializer())

def collect_files(path,file_list):
    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        if os.path.isdir(item_path):
            collect_files(item_path,file_list)
        elif item.endswith('.csv'):
            file_list.append(item_path)


def train(RL):
    total_steps = 0
    path ='sm_dataset'
    num_episodes=500
    file_list=[]
    collect_files(path,file_list)
    for episode in range(num_episodes):
        file_name=file_list[episode]
        print(episode)
        print(file_name)
        observation = env.reset(file_name)
        print(observation)
        while True:
            # if total_steps - MEMORY_SIZE > 8000: env.render()

            action = RL.choose_action(observation)
            f_action = (action-(ACTION_SPACE-1)/2)/((ACTION_SPACE-1)/8)   # convert to [-4 ~ 4] float actions
           # print(action)
           # print(f_action)                                         
           # print(type(f_action))
            # observation_, reward, done, info = env.step(np.array([f_action]))
            observation_, reward, done, info = env.step(f_action)
            print(observation_,reward, done, info)
            if done:
                break
            reward /= 10     # normalize to a range of (-1, 0). r = 0 when get upright
            # the Q target at upright state will be 0, because Q_target = r + gamma * Qmax(s', a') = 0 + gamma * 0
            # so when Q at this state is greater than 0, the agent overestimates the Q. Please refer to the final result.

            RL.store_transition(observation, action, reward, observation_)

            if total_steps > MEMORY_SIZE and total_steps % 8 == 0 : # 存满1w然后开始更新模型参数
                RL.learn()

            observation = observation_
            total_steps += 1
    return RL.q
        

q_natural = train(natural_DQN)
q_double = train(double_DQN)

model_dir = '.\model\my_net.ckpt'
saver = tf.train.Saver()
save_path = saver.save(sess, model_dir)

plt.plot(np.array(q_natural), c='r', label='natural')
plt.plot(np.array(q_double), c='b', label='double')
plt.legend(loc='best')
plt.ylabel('Q eval')
plt.xlabel('training steps')
plt.grid()
plt.savefig("test.png")
plt.show()