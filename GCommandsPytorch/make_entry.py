from __future__ import print_function
import argparse
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from gcommand_loader import GCommandLoader, AudioProcessor, spect_loader, wav_loader
import numpy as np
from model import LeNet, VGG
from train import train, test
import os
import sys

# Training settings
parser = argparse.ArgumentParser(description='ConvNets for Speech Commands Recognition')
parser.add_argument('--output_file', default='output_LeNet_ckpt.txt', help='path to the train data folder')
parser.add_argument('--test_dir', default='../test/audio', help='path to the train data folder')
parser.add_argument('--wanted_words',type=str,default='yes,no,up,down,left,right,on,off,stop,go', help='Words to use (others will be added to an unknown label)')
parser.add_argument('--label_path', default='checkpoint/labels.txt', help='path to the train data folder')
parser.add_argument('--model_path', default='checkpoint/LeNet_ckpt.t7', help='path to the train data folder')
parser.add_argument('--test_path', default='gcommands/test', help='path to the test data folder')
parser.add_argument('--test_batch_size', type=int, default=100, metavar='N', help='batch size for testing')
parser.add_argument('--arc', default='LeNet', help='network architecture: LeNet, VGG11, VGG13, VGG16, VGG19')
parser.add_argument('--cuda', default=False, help='enable CUDA')
# feature extraction options
parser.add_argument('--window_size', default=.02, help='window size for the stft')
parser.add_argument('--window_stride', default=.01, help='window stride for the stft')
parser.add_argument('--window_type', default='hamming', help='window type for the stft')
parser.add_argument('--normalize', default=True, help='boolean, wheather or not to normalize the spect')
# testing labeled data
parser.add_argument('--data_url',type=str,default='http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz',help='Location of speech training data archive on the web.')
parser.add_argument('--data_dir',type=str,default='../../gsk_train',help='Where to download the speech training data to.')
parser.add_argument('--validation_percentage', type=float, default=10.0, help='How much of the training data should be validation data.')
parser.add_argument('--testing_percentage', type=float, default=10.0, help='How much of the training data should be testing data.')
parser.add_argument('--silence_percentage', type=float, default=10.0, help='How much of the training data should be silence.')
parser.add_argument('--unknown_percentage', type=float, default=10.0, help='How much of the training data should be unknown words.')
parser.add_argument('--training_test',type=bool,default=False,help='Test Data on training set')
parser.add_argument('--analysis',type=bool,default=False,help='Test Data on training set')


args = parser.parse_args()

args.cuda = args.cuda and torch.cuda.is_available()

with open(args.label_path) as file:
    labels = [line.strip() for line in file]

model = torch.load(args.model_path)['net']
model.eval()
words = args.wanted_words.split(',')

# loading data
def evaluate_wav(wav_path, filename, model, labels, args, words):
	y, sr = wav_loader(wav_path)
	data = spect_loader(y, sr, window_size=args.window_size, window_stride=args.window_stride,
	                               window=args.window_type, normalize=args.normalize)
	if args.cuda:
	    data = data.cuda()
	data = Variable(data, volatile=True)
	data= data.unsqueeze(0)
	output = model(data)
	prediction = output.data.max(1)[1]

	output_nd = output.data.numpy()[0]

	softmax = [np.exp(i) for i in output_nd]
	max_value = max(softmax)
	max_index = softmax.index(max_value)
	label = labels[max_index]
	if label not in words:
		label = 'unknown'


	# print ('%s %s %s' % (softmax,sum(softmax), min(softmax)))
	# print ('%s,%s,%s' % (filename, labels[max_index], max_value))
	#print ('%s,%s,%s' % (filename, label, max_value))
	#
	return [filename, label, max_value]
	#print ('%s %s %s' % (labels[int(prediction)], prediction, output.data.max(1)[0]))


def _progress(text, count,  total):
        sys.stdout.write(
            '\r>> %s %.1f%%' %
            (text, float(count) / float(total) * 100.0))
        sys.stdout.flush()

if args.training_test:
	audio_processor = AudioProcessor(
      args.data_url, 
      args.data_dir, 
      args.silence_percentage,
      args.unknown_percentage,
      words, args.validation_percentage,
      args.testing_percentage)
	
	output = []
	progress = 0
	total_size = len(audio_processor.data_index['testing']) +  len(audio_processor.data_index['validation']) +  len(audio_processor.data_index['training'])
	for data_set in ['testing','validation', 'training']:
		for index, entry in enumerate(audio_processor.data_index[data_set]):
			data = evaluate_wav(entry['file'], os.path.basename(entry['file']), model, labels, args, words)
			label = entry['label']
			if label not in words:
				label = 'unknown'
			data.append(label)
			data.append(data_set)
			output.append(data)
			_progress('Analyzing', progress + index, total_size)
		progress += len(audio_processor.data_index[data_set])

	f = open(args.output_file, "w")
	if(args.analysis == True):
		f.write('fname,label,percent,truth,set\n')
		for result in output:
			f.write('%s,%s,%s,%s,%s\n' % (result[0], result[1], result[2], result[3], result[4]))
	else:
		f.write('fname,label\n')
		for result in output:
			f.write('%s,%s\n' % (result[0], result[1]))
	f.close()

else:
	output = []
	total_size = len(os.listdir(args.test_dir))
	print('# files to test:%d' % total_size)
	for index,filename in enumerate(os.listdir(args.test_dir)):
		wav_file = os.path.join(args.test_dir, filename)
		data = evaluate_wav(wav_file, filename, model, labels, args, words)
		output.append(data)
		_progress('Making Entry',index, total_size)

	f = open(args.output_file, "w")
	f.write('fname,label\n')
	for result in output:
		f.write('%s,%s\n' % (result[0], result[1]))
#