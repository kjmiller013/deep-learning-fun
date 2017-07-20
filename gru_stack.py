import os
import sys
import math
import tensorflow as tf
import gru_cell

''' Has a stack of GRU cells (possibly just one) and a linear transform into the desired output dimension (hopefully Tensorflow is smart enough not to evaluate it for the encoding part) '''
''' You need to use this if you want to have an output that isn't just the hidden state, even if there's just one GRU layer '''
''' Oh and btw, it just outputs scores rather than probabilities because apparently we can just use tf.nn.softmax_cross_entropy_with_logits() when we compute the loss '''
class GRUStack(object):
	def __init__(self, input_size, state_sizes, output_size):
		assert(len(state_sizes) > 0)
		self.my_cells = [gru_cell.GRUCell(input_size, state_sizes[0])]
		for i in xrange(1, len(state_sizes)):
			self.my_cells.append(gru_cell.GRUCell(state_sizes[i-1], state_sizes[i]))

		self.W_output = tf.Variable(tf.truncated_normal([state_sizes[-1], output_size], stddev = 1.0 / math.sqrt(state_sizes[-1])))
		self.b_output = tf.Variable(tf.zeros([output_size]))

	def __call__(self, my_input, states_in):
		assert(len(states_in) == len(self.my_cells))
		states_out = [self.my_cells[0](my_input, states_in[0])]
		for i in xrange(1, len(self.my_cells)):
			states_out.append(self.my_cells[i](states_out[i-1], states_in[i]))

		my_output = tf.matmul(states_out[-1], self.W_output) + self.b_output

		return (states_out, my_output)

	def show_me_your_trainables(self):
		my_trainables = [self.W_output]
		for my_cell in self.my_cells:
			my_trainables.extend(my_cell.show_me_your_trainables())

		return my_trainables
