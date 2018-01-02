from __future__ import print_function
import argparse
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from gcommand_loader import GCommandLoader,spect_loader
import numpy as np
from model import LeNet, VGG
from train import train, test
import os

# Training settings
parser = argparse.ArgumentParser(description='ConvNets for Speech Commands Recognition')
parser.add_argument('--output_file', default='output.txt', help='path to the train data folder')
parser.add_argument('--test_dir', default='../test/audio', help='path to the train data folder')
parser.add_argument('--wanted_words',type=str,default='yes,no,up,down,left,right,on,off,stop,go', help='Words to use (others will be added to an unknown label)')
parser.add_argument('--label_path', default='checkpoint/labels.txt', help='path to the train data folder')
parser.add_argument('--model_path', default='checkpoint/ckpt.t7', help='path to the train data folder')
parser.add_argument('--test_path', default='gcommands/test', help='path to the test data folder')
parser.add_argument('--test_batch_size', type=int, default=100, metavar='N', help='batch size for testing')
parser.add_argument('--arc', default='LeNet', help='network architecture: LeNet, VGG11, VGG13, VGG16, VGG19')
parser.add_argument('--cuda', default=False, help='enable CUDA')
# feature extraction options
parser.add_argument('--window_size', default=.02, help='window size for the stft')
parser.add_argument('--window_stride', default=.01, help='window stride for the stft')
parser.add_argument('--window_type', default='hamming', help='window type for the stft')
parser.add_argument('--normalize', default=True, help='boolean, wheather or not to normalize the spect')

args = parser.parse_args()

args.cuda = args.cuda and torch.cuda.is_available()

with open(args.label_path) as file:
    labels = [line.strip() for line in file]

model = torch.load(args.model_path)['net']
model.eval()
words = args.wanted_words.split(',')

# loading data
def evaluate_wav(wav_path, filename, model, labels, args, words):
	data = spect_loader(wav_path, window_size=args.window_size, window_stride=args.window_stride,
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


	#print ('%s %s %s' % (softmax,sum(softmax), min(softmax)))
	#print ('%s,%s,%s' % (filename, labels[max_index], max_value))
	print ('%s,%s' % (filename, label))
	#f.write('%s,%s\n' % (filename, label))

	#print ('%s %s %s' % (labels[int(prediction)], prediction, output.data.max(1)[0]))


#f = open(args.output_file, "w")
print('fname,label')
for filename in os.listdir(args.test_dir):
	wav_file = os.path.join(args.test_dir, filename)
	evaluate_wav(wav_file, filename, model, labels, args, words)

#f.close()