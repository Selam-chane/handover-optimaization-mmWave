#Main training loop for RL in a mmWave UDN environment.
import os
import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

from udn_environment import UDNEnvironment
from udn_a2c_agent import A2CAgent
import udn_utils


ITERATIONS = 10
FRAMES_PER_ITER = 1000
SKIPPING_PERIOD = 50   
UPDATE_PERIOD = 200  

LR_A = 0.002
LR_C = 0.01
GAMMA = 0.9
ENTROPY_BETA = 0.005

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_TRAIN = os.path.join(BASE_DIR, "rl_training_output.txt")


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

UDN_network = UDNEnvironment()
model = A2CAgent(
    sess,
    n_features=UDN_network.num_cells,
    n_actions=UDN_network.num_cells,
    lr_a=LR_A,
    lr_c=LR_C,
    entropy_beta=ENTROPY_BETA,
)
action_space = udn_utils.action_space()
UDN_network.allocate_bandwidth()
rate_list, HO_list, reward_list, utility_list = [], [], [], []
failed_link_list = []


for i_iter in range(ITERATIONS):
    UDN_network.reset_simulation()
    HO_per_iter, rate_per_iter, reward_per_iter = [], [], []
    rates_block, HO_cum = [], 0

  
    for iframe in range(FRAMES_PER_ITER):
        UDN_network.update_mobility()
        UDN_network.measure_snr()

        # Every 50 frames → observe state & act
        if (iframe + 1) % SKIPPING_PERIOD == 0:
            snr_vec = UDN_network.snr_matrix[0]
            state_vec = udn_utils.gen_state(snr_vec)
            s = state_vec.reshape(1,-1)           
            action = model.choose_action(s)  # Choose action
            UDN_network.user_target_cell[0] = action_space[action]
            rate, HO = UDN_network.get_reward(0)
            rates_block.append(rate)
            HO_cum += HO
            
        # Every 200 frames → compute block reward & update policy
        if (iframe + 1) % UPDATE_PERIOD == 0:
            block_reward = udn_utils.cal_reward(rates_block, HO_cum)
            s_next = state_vec.reshape(-1, 1)
            v_s_next = model.target_v(s_next)
            td_target = block_reward + GAMMA * v_s_next
            
            # Policy update
            s_train = state_vec.reshape(1, -1)
            feed_dict = {
                model.s: s_train,
                model.a: np.vstack([action]),
                model.td_target: np.vstack([td_target]),
            }
            model.learn(feed_dict)
            rate_per_iter.append(min(rates_block) if rates_block else 0.0)
            HO_per_iter.append(HO_cum)
            reward_per_iter.append(block_reward)
            rates_block, HO_cum = [], 0


    last_rate = rate_per_iter[-1] if rate_per_iter else 0.0
    #last_rate = min(rate_per_iter)
    total_HO = sum(HO_per_iter)
    mean_reward = np.mean(reward_per_iter) if reward_per_iter else 0.0
    
    


    rate_list.append(last_rate)
    HO_list.append(total_HO)
    reward_list.append(mean_reward)
    
    


    print(f"\nIteration {i_iter}")
    print(f"Throughput: {last_rate:.3f} Mbps")
    print(f"Handover Rate: {total_HO/10}")
    print(f"Mean reward: {mean_reward:.4f}")
    

    # Periodic file logging
    if (i_iter + 1) % 5 == 0:
        with open(LOG_TRAIN, "a+", encoding="utf-8") as f:
            for i in range(len(rate_list)):
                print(
                    "Reward: %.4f, Handover Rate: %.4f, Throughput: %.4f"
                    % (
                        reward_list[i],
                        HO_list[i]/10,
                        rate_list[i],
                    ),
                    file=f,
                )
       
        rate_list, HO_list, reward_list, utility_list, failed_link_list = [], [], [], [], []



