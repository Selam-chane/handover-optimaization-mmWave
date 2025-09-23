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
REPORT_PERIOD = 50   
UPDATE_PERIOD = 200  

LR_A = 0.002
LR_C = 0.01
GAMMA = 0.9
ENTROPY_BETA = 0.005

LOG_TRAIN = os.path.expanduser( r"D:/RL codes/MAIN_LOOP_TEST.txt")

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
        if (iframe + 1) % REPORT_PERIOD == 0:
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
    total_HO = sum(HO_per_iter)
    mean_reward = np.mean(reward_per_iter) if reward_per_iter else 0.0
    iter_utility = 0.02 * last_rate - 0.1 * total_HO
    failed_links_no = np.sum(np.array(rate_per_iter) < 50.0)


    rate_list.append(last_rate)
    HO_list.append(total_HO)
    reward_list.append(mean_reward)
    utility_list.append(iter_utility)
    failed_link_list.append(failed_links_no)


    print(f"\nIteration {i_iter}")
    print(f"Rate (last block min): {last_rate:.3f} Mbps")
    print(f"HO count: {total_HO}")
    print(f"Mean reward: {mean_reward:.4f}")
    print(f"Block rewards: {np.array(reward_per_iter)}")

    # Periodic file logging
    if (i_iter + 1) % 5 == 0:
        with open(LOG_TRAIN, "a+", encoding="utf-8") as f:
            for i in range(len(rate_list)):
                print(
                    "Reward: %.4f, HO: %.4f, QoE: %.4f, Utility: %.4f, FailedLinks: %.4f"
                    % (
                        reward_list[i],
                        HO_list[i],
                        rate_list[i],
                        utility_list[i],
                        failed_link_list[i],
                    ),
                    file=f,
                )
       
        rate_list, HO_list, reward_list, utility_list, failed_link_list = [], [], [], [], []



