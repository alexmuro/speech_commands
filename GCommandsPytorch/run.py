from __future__ import print_function
import argparse
import torch
import torch.optim as optim
from gcommand_loader import GCommandLoader, AudioProcessor, spect_loader
import numpy as np
from model import LeNet, VGG
from train import train, test
import os

# Training settings
parser = argparse.ArgumentParser(description='ConvNets for Speech Commands Recognition')
parser.add_argument('--checkpoint_dir', default='checkpoint', help='path to the train data folder')
parser.add_argument('--batch_size', type=int, default=100, metavar='N', help='training and valid batch size')
parser.add_argument('--test_batch_size', type=int, default=100, metavar='N', help='batch size for testing')
parser.add_argument('--wanted_words',type=str,default='yes,no,up,down,left,right,on,off,stop,go', help='Words to use (others will be added to an unknown label)')
parser.add_argument('--arc', default='LeNet', help='network architecture: LeNet, VGG11, VGG13, VGG16, VGG19')
parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum, for SGD only')
parser.add_argument('--optimizer', default='adam', help='optimization method: sgd | adam')
parser.add_argument('--cuda', default=True, help='enable CUDA')
parser.add_argument('--seed', type=int, default=1234, metavar='S', help='random seed')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',help='how many batches to wait before logging training status')
parser.add_argument('--patience', type=int, default=5, metavar='N',help='how many epochs of no loss improvement should we wait before stop training')

# feature extraction options
parser.add_argument('--window_size', default=.02, help='window size for the stft')
parser.add_argument('--window_stride', default=.01, help='window stride for the stft')
parser.add_argument('--window_type', default='hamming', help='window type for the stft')
parser.add_argument('--normalize', default=True, help='boolean, wheather or not to normalize the spect')
parser.add_argument('--background_volume', type=float, default=0.1, help='How loud the background noise should be, between 0 and 1.')
parser.add_argument('--background_frequency',type=float, default=0.8, help='How many of the training samples have background noise mixed in.')
parser.add_argument('--validation_percentage', type=float, default=10.0, help='How much of the training data should be validation data.')
parser.add_argument('--testing_percentage', type=float, default=10.0, help='How much of the training data should be testing data.')
parser.add_argument('--silence_percentage', type=float, default=10.0, help='How much of the training data should be silence.')
parser.add_argument('--unknown_percentage', type=float, default=10.0, help='How much of the training data should be unknown words.')
parser.add_argument('--data_url',type=str,default='http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz',help='Location of speech training data archive on the web.')
parser.add_argument('--data_dir',type=str,default='../../gsk_train',help='Where to download the speech training data to.')
args = parser.parse_args()

args.cuda = args.cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

wanted_words = args.wanted_words.split(',')
audio_processor = AudioProcessor(
      args.data_url, args.data_dir, args.silence_percentage,
      args.unknown_percentage,
      wanted_words, args.validation_percentage,
      args.testing_percentage)


## test loading script
sample = audio_processor.data_index['training'][0]
spect = spect_loader(sample['file'], args.window_size, args.window_stride, 'hamming', True, 101)
print(spect)
print(spect.mul(0))
#loading data
train_dataset = GCommandLoader(
  audio_processor.data_index['training'],
  classes=audio_processor.words_list,
  class_to_idx=audio_processor.word_to_index,
  window_size=args.window_size, 
  window_stride=args.window_stride,
  window_type=args.window_type,
  normalize=args.normalize)

train_loader = torch.utils.data.DataLoader(
  train_dataset, 
  batch_size=args.batch_size, 
  shuffle=True,
  num_workers=20, 
  pin_memory=args.cuda, 
  sampler=None)

valid_dataset = GCommandLoader(
  audio_processor.data_index['validation'],
  classes=audio_processor.words_list,
  class_to_idx=audio_processor.word_to_index,
  window_size=args.window_size, 
  window_stride=args.window_stride,
  window_type=args.window_type,
  normalize=args.normalize)

valid_loader = torch.utils.data.DataLoader(
    valid_dataset, batch_size=args.batch_size, shuffle=None,
    num_workers=20, pin_memory=args.cuda, sampler=None)

test_dataset = GCommandLoader(
  audio_processor.data_index['testing'],
  classes=audio_processor.words_list,
  class_to_idx=audio_processor.word_to_index, 
  window_size=args.window_size, 
  window_stride=args.window_stride,
  window_type=args.window_type,
  normalize=args.normalize)

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=args.test_batch_size, shuffle=None,
    num_workers=20, pin_memory=args.cuda, sampler=None)

# build model
if args.arc == 'LeNet':
    model = LeNet()
elif args.arc.startswith('VGG'):
    model = VGG(args.arc)
else:
    model = LeNet()

if args.cuda:
    print('Using CUDA with {0} GPUs'.format(torch.cuda.device_count()))
    model = torch.nn.DataParallel(model).cuda()

# define optimizer
if args.optimizer.lower() == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
elif args.optimizer.lower() == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
else:
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

best_valid_loss = np.inf
iteration = 0
epoch = 1

if not os.path.isdir(args.checkpoint_dir):
        os.mkdir(args.checkpoint_dir)
f = open(args.checkpoint_dir + "/labels.txt", "w")
f.write('\n'.join(train_dataset.classes))
f.close()

# trainint with early stopping
while (epoch < args.epochs + 1) and (iteration < args.patience):
    train(train_loader, model, optimizer, epoch, args.cuda, args.log_interval)
    valid_loss = test(valid_loader, model, args.cuda)
    if valid_loss > best_valid_loss:
        iteration += 1
        print('Loss was not improved, iteration {0}'.format(str(iteration)))
    else:
        print('Saving model...')
        iteration = 0
        best_valid_loss = valid_loss
        state = {
            'net': model.module if args.cuda else model,
            'acc': valid_loss,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, args.checkpoint_dir + '/' + args.arc + '_ckpt.t7')
    epoch += 1

# test model
test(test_loader, model, args.cuda)
