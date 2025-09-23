
#A2C (Advantage Actor-Critic) Agent in TensorFlow v1

import numpy as np
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()


### RL ALGORITHM A2C
class A2CAgent(object):   
    def __init__(
            self,  
            sess,
            n_actions,
            n_features,
            lr_a=0.005,
            lr_c=0.01,
            entropy_beta=0.01
    ):
        self.sess = sess
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr_a = lr_a
        self.lr_c = lr_c
        self.entroy_beta = entropy_beta
        
        # Optimizers
        self.opt_actor = tf.train.AdamOptimizer(self.lr_a) 
        self.opt_critic = tf.train.AdamOptimizer(self.lr_c) 
        
        
       
        with tf.name_scope('inputs'):
            self.s = tf.placeholder(tf.float32, [1,self.n_features], "state")
            self.a = tf.placeholder(tf.int32, [None,1], "action")
            self.td_target = tf.placeholder(tf.float32, [None, 1], "td_target")
            
        # Build networks
        self.acts_prob, self.v, self.a_params, self.c_params = self._build_net()
        
        
        
        # Critic loss
        with tf.name_scope('TD_error'):
            self.td_error = tf.subtract(self.td_target, self.v, name='TD_error')

        with tf.name_scope('c_loss'):
            self.c_loss = tf.reduce_mean(tf.square(self.td_error))
            
            
        
        # Actor loss
        with tf.name_scope('a_loss'):
            log_prob = tf.reduce_sum(tf.log(self.acts_prob + 1e-5) * tf.one_hot(self.a, self.n_actions, dtype=tf.float32),
                                     axis=1, keepdims=True)
            
            # Policy gradient objective
            exp_v = log_prob * tf.stop_gradient(self.td_error)
            
            # Entropy regularization
            entropy = -tf.reduce_sum(self.acts_prob * tf.log(self.acts_prob + 1e-5), axis=1,
                                     keepdims=True)  # encourage exploration
            self.exp_v = self.entroy_beta * entropy + exp_v
            self.a_loss = tf.reduce_mean(-self.exp_v)
            
            
       
        # Training      
        with tf.name_scope('compute_grads'):
            self.a_grads = tf.gradients(self.a_loss, self.a_params)
            self.c_grads = tf.gradients(self.c_loss, self.c_params)

        with tf.name_scope('c_train'):
            self.c_train_op = self.opt_critic.apply_gradients(zip(self.c_grads, self.c_params))

        with tf.name_scope('a_train'):
            self.a_train_op = self.opt_actor.apply_gradients(zip(self.a_grads, self.a_params))
            
        self.sess.run(tf.global_variables_initializer())
    tf.reset_default_graph()


   
    def _build_net(self):
        w_init = tf.random_normal_initializer(0., .1)
        b_init = tf.constant_initializer(0.1)
        
        
        # ----- Critic -----
        with tf.variable_scope('Critic'):
  
            l_c1 = tf.layers.dense(
                inputs=self.s,
                units=32,
                activation=tf.nn.tanh,
                kernel_initializer=w_init,
                bias_initializer=b_init,
                name='l_c1'
            )
            
            v = tf.layers.dense(
                inputs=l_c1,
                units=1,
                kernel_initializer=w_init,
                bias_initializer=b_init,
                name='V'
            )  # state value

        # ----- Actor -----
        with tf.variable_scope('Actor',reuse=tf.AUTO_REUSE):
            l_a1 = tf.layers.dense(
                inputs=self.s,
                units=32,
                activation=tf.nn.tanh,
                kernel_initializer=w_init, 
                bias_initializer=b_init,  
                name='l_a1'
            )
            
            acts_prob = tf.layers.dense(
                inputs=l_a1,
                units=self.n_actions,  
                activation=tf.nn.softmax,  
                kernel_initializer=w_init,  
                name='acts_prob'
            )
        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Critic')

        return acts_prob, v, a_params, c_params    
    

    # Action selection  
    def choose_action(self, st):
        st=st.reshape(1,-1)
        probs = self.sess.run(self.acts_prob, feed_dict={self.s: st})  # get probabilities for all actions
        probs_1d = probs.reshape(-1,1)
        a = np.random.choice(np.arange(probs.shape[1]), p=probs_1d.ravel())
        return a


    # Learning
    def learn(self, feed_dict):
        self.sess.run([self.a_train_op, self.c_train_op], feed_dict=feed_dict)


    def target_v(self, s):
        s =s.reshape(1,-1)
        v = self.sess.run(self.v, {self.s: s})
        return v



