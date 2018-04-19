# -*- coding = utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import read_data
import model
import math
import tensorflow as tf

NUM_TRAIN_EXAMPLES = read_data.NUM_TRAIN_EXAMPLES
NUM_VALIDATION_EXAMPLES = read_data.NUM_VALIDATION_EXAMPLES
EVAL_DATA_DIR = 'tmp/eval_data'
BATCH_SIZE = model.BATCH_SIZE

def evaluate(data_set,checkpoint_dir = 'tmp/train_data'):
	with tf.Graph().as_default():
		if data_set == 'validation':
			num_examples = NUM_VALIDATION_EXAMPLES
		elif data_set == 'train':
			num_examples = NUM_TRAIN_EXAMPLES
		else:
			raise ValueError('data_set should be one of \'train\', \'validation\'')

		images, labels = read_data.inputs(data_set=data_set, batch_size=BATCH_SIZE, num_epochs=None)
		logits = model.inference(images)
		accuracy_curr_batch = model.evaluation(logits, labels)

		mov_avg_obj = tf.train.ExponentialMovingAverage(model.MOVING_AVERAGE_DECAY)
		variables_to_restore = mov_avg_obj.variables_to_restore()
		saver = tf.train.Saver(variables_to_restore)

		with tf.Session() as sess:
			ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
			if ckpt and ckpt.model_checkpoint_path:
				saver.restore(sess, ckpt.model_checkpoint_path)
			else:
				print('No checkpoint file found at %s' % checkpoint_dir)
				return

			coord = tf.train.Coordinator()

			try:
				threads = []
				for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
					threads.extend(qr.create_threads(sess, coord, daemon=True, start=True))

				num_iter = int(math.ceil(num_examples / BATCH_SIZE))
				step = 0
				acc_full_epoch = 0
				while step < num_iter and not coord.should_stop():
					acc_batch_val = sess.run(accuracy_curr_batch)
					acc_full_epoch += acc_batch_val
					step += 1

				acc_full_epoch /= num_iter
				tf.summary.scalar('validation_accuracy', acc_full_epoch)
				summary_op = tf.summary.merge_all()
				summary_writer = tf.summary.FileWriter(EVAL_DATA_DIR)
				summary_str = sess.run(summary_op)
				summary_writer.add_summary(summary_str, step)

				print('Accuracy on full %s dataset = %.1f' % (data_set, acc_full_epoch))


			except Exception as e:
				coord.request_stop(e)

			coord.request_stop()

			coord.join(threads)

def main(argv):
	data_set = argv[1]
	evaluate(data_set)


if __name__ == '__main__':
	tf.app.run()
