import os
import sys
import math
import tensorflow as tf

''' includes bias terms, which will be initialized to zero but won't be regularized '''
''' Also, any linear transformation will be represented as tf.matmul(input, W) because it seems like that's how people like to do it around here '''
class GRUCell(object):
	def __init__(self, input_size, state_size):
		self.W_reset_taking_input = tf.Variable(tf.truncated_normal([input_size, state_size], stddev = 1.0 / math.sqrt(input_size + state_size)))
		self.W_reset_taking_state = tf.Variable(tf.truncated_normal([state_size, state_size], stddev = 1.0 / math.sqrt(input_size + state_size)))
		self.b_reset = tf.Variable(tf.zeros([state_size]))
		self.W_forget_taking_input = tf.Variable(tf.truncated_normal([input_size, state_size], stddev = 1.0 / math.sqrt(input_size + state_size)))
		self.W_forget_taking_state = tf.Variable(tf.truncated_normal([state_size, state_size], stddev = 1.0 / math.sqrt(input_size + state_size)))
		self.b_forget = tf.Variable(tf.zeros([state_size]))
		self.W_transform_taking_input = tf.Variable(tf.truncated_normal([input_size, state_size], stddev = 1.0 / math.sqrt(input_size + state_size)))
		self.W_transform_taking_state = tf.Variable(tf.truncated_normal([state_size, state_size], stddev = 1.0 / math.sqrt(input_size + state_size)))
		self.b_transform = tf.Variable(tf.zeros([state_size]))

	def __call__(self, my_input, state_in):
		reset_gate = tf.sigmoid(tf.matmul(my_input, self.W_reset_taking_input) + tf.matmul(state_in, self.W_reset_taking_state) + self.b_reset)
		forget_gate = tf.sigmoid(tf.matmul(my_input, self.W_forget_taking_input) + tf.matmul(state_in, self.W_forget_taking_state) + self.b_forget)
		state_increment = tf.tanh(tf.matmul(my_input, self.W_transform_taking_input) + tf.matmul(reset_gate * state_in, self.W_transform_taking_state) + self.b_transform)
		state_out = (1.0 - forget_gate) * state_in + forget_gate * state_increment
		return state_out

	def show_me_your_trainables(self):
		return [self.W_reset_taking_input, self.W_reset_taking_state, self.W_forget_taking_input, self.W_forget_taking_state, self.W_transform_taking_input, self.W_transform_taking_state]
