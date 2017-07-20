import os
import sys
import numpy as np
import random
import tensorflow as tf
import gru_stack

SEQUENCE_LENGTH = 40

def one_hot_fuckyou(i, N):
	a = np.zeros(N, dtype='float32')
	a[i] = 1.0
	return a

''' this will be a list of one-hot vectors '''
def one_hot_sequence_fuckyou(sequence, num_possible_inputs):
	return [one_hot_fuckyou(i, num_possible_inputs + 1) for i in sequence] + [one_hot_fuckyou(num_possible_inputs, num_possible_inputs + 1)]

def one_hot_sequence(sequence, num_possible_inputs):
	meow = one_hot_sequence_fuckyou(sequence, num_possible_inputs)
	return [np.vstack([purr]) for purr in meow] 

def make_init_states_in(state_sizes):
	return [np.zeros((1, state_size), dtype='float32') for state_size in state_sizes]

def generate_one_sentence(my_gru_stack, input_size, state_sizes, num_chars_to_generate=100):
	init_state_in_placeholders = [tf.placeholder(tf.float32, shape=[1,state_size]) for state_size in state_sizes]
	first_char_placeholder = tf.placeholder(tf.float32, shape=[1,input_size])
	outputs = []
	cur_states = init_state_in_placeholders
	cur_input = first_char_placeholder
	for i in xrange(num_chars_to_generate):
		(cur_states, cur_input) = my_gru_stack(cur_input, cur_states)
		cur_input = tf.one_hot(tf.argmax(cur_input, axis=1), input_size)
		outputs.append(cur_input)

	return (tf.stack(outputs), init_state_in_placeholders, first_char_placeholder)

''' Builds the computation graph for handling a batch of sequences '''
''' Each sequence gets a separate placeholder, unfortunately '''
''' return loss function, sequence_placeholders, and init_state_in_placeholders '''
def build_computation_graph_one_batch(my_gru_stack, batch_sequence_lengths, input_size, state_sizes, regularization_weight):
	sequence_placeholders = [[tf.placeholder(tf.float32, shape=[1,input_size]) for j in xrange(my_length)] for my_length in batch_sequence_lengths]
	init_state_in_placeholders = [tf.placeholder(tf.float32, shape=[1,state_size]) for state_size in state_sizes]
	trainables = my_gru_stack.show_me_your_trainables()
	loss = 0.0
	for trainable in trainables:
		loss += regularization_weight * tf.reduce_sum(trainable * trainable)

	for item_placeholders in sequence_placeholders:
		cur_states = init_state_in_placeholders
		cur_output = None
		for item_placeholder in item_placeholders:
			(cur_states, cur_output) = my_gru_stack(item_placeholder, cur_states)

		labels = tf.stack(item_placeholders)
		logits = [cur_output]
		for j in xrange(1, len(item_placeholders)):
			(cur_states, cur_output) = my_gru_stack(item_placeholders[j-1], cur_states)
			logits.append(cur_output)

		logits = tf.stack(logits)
		loss += tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)) / (1.0 * len(batch_sequence_lengths))

	return (loss, sequence_placeholders, init_state_in_placeholders)

def generation_helper(sess, outputs, gen_feed_dict, ix_to_char):
	my_outputs = sess.run(outputs, feed_dict = gen_feed_dict)
	meow = ''.join([ix_to_char[ix] for ix in np.argmax(np.reshape(my_outputs, (my_outputs.shape[0], my_outputs.shape[2])), axis=1)])
	print('MEOW:')
	print(meow)

''' sequences is a list of lists of nonnegative ints '''
''' caller is not expected to do anything to handle end-of-sequence behavior '''
''' returns a trained GRUStack object '''
def train_gru_sa(sess, sequences, num_possible_inputs, ix_to_char, state_sizes=[128, 128, 128], regularization_weight=1e-3, batch_size=1, learning_rate=0.01, num_iters = 1000, logging_frequency = 1):
	sequence_lengths = [len(sequence) + 1 for sequence in sequences] #just in case we decide to go back to representing the sequences as matrices
	print(str(len(sequences)) + ' sequences')
	init_states_in = make_init_states_in(state_sizes)
	
	input_size = num_possible_inputs + 1
	my_gru_stack = gru_stack.GRUStack(input_size, state_sizes, input_size)

	init = tf.global_variables_initializer()
	sess.run(init)

	print('building generation graph...')
	(outputs, gen_init_state_in_placeholders, first_char_placeholder) = generate_one_sentence(my_gru_stack, input_size, state_sizes)
	print('done building generation graph')
	print('building gen_feed_dict...')
	gen_feed_dict = {}
	for l in xrange(len(init_states_in)):
		gen_feed_dict[gen_init_state_in_placeholders[l]] = np.random.randn(1, state_sizes[l])

	my_first_char = np.zeros((1, input_size), dtype='float32')
	my_first_char[:,10] = 1.0
	gen_feed_dict[first_char_placeholder] = my_first_char
	print('done building gen_feed_dict')

	generation_helper(sess, outputs, gen_feed_dict, ix_to_char)

	print('building computation graph...')
	(loss, sequence_placeholders, init_state_in_placeholders) = build_computation_graph_one_batch(my_gru_stack, [SEQUENCE_LENGTH + 1] * batch_size, input_size, state_sizes, regularization_weight)
	print('done building computation graph')		
	
	print('train_step...')
	my_opt = tf.train.GradientDescentOptimizer(learning_rate)
	gvs = my_opt.compute_gradients(loss)
	capped_gvs = [(tf.clip_by_value(grad, -1.0, 1.0), var) for grad, var in gvs]
	train_step = my_opt.apply_gradients(capped_gvs)
	print('done train_step')

	smoothed_loss = None
	loss_memory = 0.9
	
	for t in xrange(num_iters):
		batch_indices = random.sample(range(len(sequences)), batch_size)
		batch_sequence_lengths = [sequence_lengths[i] for i in batch_indices]
		print('encoding data...')
		batch_sequences = [one_hot_sequence(sequences[i], num_possible_inputs) for i in batch_indices]
		print('done encoding data')

		print('building feed_dict...')
		feed_dict = {}
		for i in xrange(batch_size):
			my_sequence = batch_sequences[i]
			print(len(my_sequence))
			for j in xrange(len(my_sequence)):
				feed_dict[sequence_placeholders[i][j]] = my_sequence[j]

		for l in xrange(len(init_states_in)):
			feed_dict[init_state_in_placeholders[l]] = init_states_in[l]

		for k in feed_dict.keys():
			assert(feed_dict[k].dtype == np.dtype('float32'))
			assert(feed_dict[k].dtype != np.dtype('S32'))

		print('done building feed_dict')

#		import pdb
#		pdb.set_trace()
		sess.run(train_step, feed_dict = feed_dict)

#		import pdb
#		pdb.set_trace()

		if t % logging_frequency == 0:
			if smoothed_loss == None:
				smoothed_loss = sess.run(loss, feed_dict = feed_dict)
			else:
				smoothed_loss = (1.0 - loss_memory) * sess.run(loss, feed_dict = feed_dict) + loss_memory * smoothed_loss

			print(str(t) + ': ' + str(smoothed_loss))

		print('building gen_feed_dict...')
		gen_feed_dict = {}
		for l in xrange(len(init_states_in)):
			gen_feed_dict[gen_init_state_in_placeholders[l]] = 20.0 * np.random.randn(1, state_sizes[l])

		my_first_char = np.zeros((1, input_size), dtype='float32')
		my_first_char[:,10] = 1.0
		gen_feed_dict[first_char_placeholder] = my_first_char
		print('done building gen_feed_dict')
		generation_helper(sess, outputs, gen_feed_dict, ix_to_char)

	return my_gru_stack

if __name__ == '__main__':
	# data I/O
	data = open('input.txt', 'r').read() # should be simple plain text file
	data = data.lower()
	data = data.replace('.', '')
	data = data.replace(',', '')
	data = data.replace('"', '')
	data = data.replace('\'', '')
	data = data.replace('!', '')
	data = data.replace('?', '')
	data = data.replace(':', '')
	data = data.replace(';', '')
	data = data.replace('(', '')
	data = data.replace(')', '')
	chars = list(set(data))
	chars = [c for c in chars if c != '\n']
	data_size, vocab_size = len(data), len(chars)
	print 'data has %d characters, %d unique.' % (data_size, vocab_size)
	char_to_ix = { ch:i for i,ch in enumerate(chars)}
	ix_to_char = { i:ch for i,ch in enumerate(chars)}
	ix_to_char[vocab_size] = '\n'
	#ix_to_char[char_to_ix[' ']] = '_'
	print(ix_to_char)
	sequences = data.split('\n')
	sequences = [sequence[:SEQUENCE_LENGTH] for sequence in sequences if len(sequence) >= SEQUENCE_LENGTH]
	print('%d sequences'%(len(sequences)))
	sequences = [[char_to_ix[c] for c in sequence] for sequence in sequences]

	config = tf.ConfigProto(device_count = {'GPU': 0})
	
	sess = tf.Session(config=config)
	my_gru_stack = train_gru_sa(sess, sequences, vocab_size, ix_to_char)

	import pdb
	pdb.set_trace()
