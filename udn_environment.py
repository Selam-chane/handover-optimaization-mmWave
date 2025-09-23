import numpy as np
import random

class UDNEnvironment:
   
    #Simulation environment for a mmWave Ultra-Dense Network (UDN).

    def __init__(self,
                 area_radius=500,
                 num_cells=50,
                 bs_tx_power=16,          # dBm
                 num_users=100,
                 noise_psd=-174,          # dBm/Hz (thermal noise density)
                 channel_model='38901',   # 3GPP TR 38.901 version 14.0.0 : Pathloss models UMi - Street Canyon
                 carrier_freq=28e9,       # Hz (28 GHz, FR-2 Band n261)
                 subframe_length=1e-3,    # seconds
                 frame_length=10e-3,      # seconds
                 bandwidth=50e6,          # Hz (50 MHz)
                 dl_mimo=32,
                 tx_gain=8,               # dB
                 rx_gain=8,               # dB
                 learning_window=10000,
                 rrc_states=('Idle', 'Connected', 'Inactive'),
                 rrc_state_prob=np.array([2, 6, 2], dtype=np.float32),
                 seed=42):

        # Set seed for reproducibility
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        # System parameters
        self.bs_tx_power = bs_tx_power
        self.area_radius = 200
        self.bandwidth = bandwidth
        self.channel_model = channel_model
        self.carrier_freq = carrier_freq
        self.frame_duration = round(frame_length, 3)
        self.noise_psd = noise_psd
        self.dl_mimo = dl_mimo
        self.tx_gain = tx_gain
        self.rx_gain = rx_gain
        self.num_users = num_users
        self.num_cells = 25
        self.learning_window = round(learning_window * self.frame_duration, 3)

        # System clock
        self.sim_time = 0.0
        self.frame_index = 0

        # RRC states
        self.rrc_states = rrc_states
        self.rrc_state_prob = rrc_state_prob / np.sum(rrc_state_prob)
        self.user_rrc_state = np.random.choice(self.rrc_states, self.num_users, p=self.rrc_state_prob)
        self.user_rrc_state[0] = 'Connected'  # ensure at least one connected user

        # User positions and mobility parameters
        self.user_displacement = np.zeros(self.num_users)
        self.user_positions = np.random.uniform(-self.area_radius, self.area_radius, [self.num_users, 2])
        self.user_serving_cell = np.random.randint(0, self.num_cells, self.num_users)
        self.user_target_cell = np.random.randint(0, self.num_cells, self.num_users)
        self.user_speeds = np.random.uniform(10, 100, self.num_users)  # constant speed for all users
        self.user_directions = np.random.uniform(-180, 180, self.num_users)
        self.cell_ids = np.arange(num_cells)
        self.cell_positions = np.zeros((len(self.cell_ids), 2))
        for i in range(num_cells):
            row, col = divmod(i, 5)
            self.cell_positions[i, 0] = col * 100 - 200
            self.cell_positions[i, 1] = 200 - row * 100
        self.mgl= np.array([1.5e-3, 3e-3, 3.5e-3, 4e-3, 5.5e-3, 6e-3])
        self.mgrp_choices = np.array([20e-3, 40e-3, 80e-3, 160e-3])
        self.gap_offset = 0
        self.user_mgrps = np.random.choice(self.mgrp_choices, self.num_users)
        self.snr_matrix = np.zeros((self.num_users, self.num_cells))
        self.reported_snr = []
        self.user_bandwidth = np.zeros(self.num_users)
        self.reported_state = np.zeros((self.num_users, self.num_cells))
        self.handover_count = 0
        self.shadowing_variance = np.random.normal(0, 8.2)

   
    def regenerate_rrc_states(self):    
        self.user_rrc_state = np.random.choice(self.rrc_states, self.num_users, p=self.rrc_state_prob)


    def allocate_bandwidth(self):
        connected_users = np.where(self.user_rrc_state == 'Connected')[0]
        if len(connected_users) > 0:
            self.user_bandwidth[connected_users] = self.bandwidth / len(connected_users)
        else:
            self.user_bandwidth[:] = 0
        return self.user_bandwidth

  
    # Channel modeling
    def calculate_Rx(self, user_idx):
        rx_powers = []
        distances = []
        if self.channel_model == '38901':
            for cell_idx in range(self.num_cells):
                distance = np.linalg.norm(self.cell_positions[cell_idx] - self.user_positions[user_idx])
                path_loss = 61.34 + 31.9 * np.log10(round(distance, 3))  
                total_loss = path_loss + self.shadowing_variance
                rx_power = self.bs_tx_power + self.tx_gain - total_loss + self.rx_gain
                rx_powers.append(rx_power)
                distances.append(distance)
        return rx_powers, distances


    def measure_snr(self):
        if self.sim_time == 0:
            self.frame_index = 0
        else:
            if round((self.sim_time - self.frame_duration) * 1000, 3) % (self.frame_duration * 1000) == 0:
                self.frame_index += 1
        connected_users = np.where(self.user_rrc_state == 'Connected')[0]
        for ue in connected_users:
            mgrp_frames = int((self.user_mgrps[ue] * 1000) / 10)
            if self.gap_offset == 0 and self.frame_index % mgrp_frames == 0:
                rx_powers, _ = self.calculate_Rx(ue)
                noise_power = self.noise_psd + 10 * np.log10(self.user_bandwidth[ue])
                self.snr_matrix[ue, :] = np.array(rx_powers) - noise_power
        self.sim_time += self.frame_duration
        return self.snr_matrix


    def update_mobility(self):
        displacement = self.user_speeds * self.frame_duration
        delta_x = displacement * np.cos(np.deg2rad(self.user_directions))
        delta_y = displacement * np.sin(np.deg2rad(self.user_directions))
        self.user_positions[:, 0] += delta_x
        self.user_positions[:, 1] += delta_y
        R = self.area_radius
        for axis in [0, 1]:
            lower_idx = np.where(self.user_positions[:, axis] < -R)
            self.user_positions[lower_idx, axis] = -2 * R - self.user_positions[lower_idx, axis]
            self.user_directions[lower_idx] = 180 - self.user_directions[lower_idx]

            upper_idx = np.where(self.user_positions[:, axis] > R)
            self.user_positions[upper_idx, axis] = 2 * R - self.user_positions[upper_idx, axis]
            self.user_directions[upper_idx] = 180 - self.user_directions[upper_idx]


    def reset_simulation(self):
        self.sim_time = 0.0
        self.frame_index = 0
        self.snr_matrix.fill(0.0)
        self.reported_state.fill(0.0)
        self.reported_snr = []

    def trigger_event_A2(self, user_idx):
        self.reported_snr = self.snr_matrix[user_idx, :]
        return self.reported_snr

    def store_state(self):
        connected_users = np.where(self.user_rrc_state == 'Connected')[0]
        for ue in connected_users:
            self.trigger_event_A2(ue)
            self.reported_state[ue, :] = self.reported_snr

    def get_state(self):
        reported_users = []
        connected_users = np.where(self.user_rrc_state == 'Connected')[0]
        for ue in connected_users:
            if not np.allclose(self.reported_state[ue], np.zeros_like(self.reported_state[ue])):
                reported_users.append(ue)
        return reported_users

    def get_reward(self, user_idx):
        handover_flag = int(self.user_target_cell[user_idx] != self.user_serving_cell[user_idx])
        snr_value = self.snr_matrix[user_idx, self.user_target_cell[user_idx]]
        rate_bps = self.user_bandwidth[user_idx] * np.log2(1 + 10 ** (snr_value / 10)) * self.dl_mimo
        rate_mbps = rate_bps * 1e-6
        self.user_serving_cell[user_idx] = self.user_target_cell[user_idx]
        return rate_mbps, handover_flag
