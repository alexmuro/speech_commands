import torch.utils.data as data

import os
import os.path
import torch
import sys
import tarfile
import random
import re
import hashlib
import math

from six.moves import urllib
from six.moves import xrange

from tensorflow.python.platform import gfile
from tensorflow.python.util import compat

import librosa
import numpy as np

AUDIO_EXTENSIONS = [
    '.wav', '.WAV',
]
MAX_NUM_WAVS_PER_CLASS = 2**27 - 1  # ~134M
SILENCE_LABEL = 'silence'
SILENCE_INDEX = 0
UNKNOWN_WORD_LABEL = 'unknown'
UNKNOWN_WORD_INDEX = 1
BACKGROUND_NOISE_DIR_NAME = '_background_noise_'
RANDOM_SEED = 59185

def prepare_words_list(wanted_words):
  """Prepends common tokens to the custom word list.

  Args:
    wanted_words: List of strings containing the custom words.

  Returns:
    List with the standard silence and unknown tokens added.
  """
  return [SILENCE_LABEL, UNKNOWN_WORD_LABEL] + wanted_words

def which_set(filename, validation_percentage, testing_percentage):
  """Determines which data partition the file should belong to.

  We want to keep files in the same training, validation, or testing sets even
  if new ones are added over time. This makes it less likely that testing
  samples will accidentally be reused in training when long runs are restarted
  for example. To keep this stability, a hash of the filename is taken and used
  to determine which set it should belong to. This determination only depends on
  the name and the set proportions, so it won't change as other files are added.

  It's also useful to associate particular files as related (for example words
  spoken by the same person), so anything after '_nohash_' in a filename is
  ignored for set determination. This ensures that 'bobby_nohash_0.wav' and
  'bobby_nohash_1.wav' are always in the same set, for example.

  Args:
    filename: File path of the data sample.
    validation_percentage: How much of the data set to use for validation.
    testing_percentage: How much of the data set to use for testing.

  Returns:
    String, one of 'training', 'validation', or 'testing'.
  """
  base_name = os.path.basename(filename)
  # We want to ignore anything after '_nohash_' in the file name when
  # deciding which set to put a wav in, so the data set creator has a way of
  # grouping wavs that are close variations of each other.
  hash_name = re.sub(r'_nohash_.*$', '', base_name)
  # This looks a bit magical, but we need to decide whether this file should
  # go into the training, testing, or validation sets, and we want to keep
  # existing files in the same set even if more files are subsequently
  # added.
  # To do that, we need a stable way of deciding based on just the file name
  # itself, so we do a hash of that and then use that to generate a
  # probability value that we use to assign it.
  hash_name_hashed = hashlib.sha1(compat.as_bytes(hash_name)).hexdigest()
  percentage_hash = ((int(hash_name_hashed, 16) %
                      (MAX_NUM_WAVS_PER_CLASS + 1)) *
                     (100.0 / MAX_NUM_WAVS_PER_CLASS))
  if percentage_hash < validation_percentage:
    result = 'validation'
  elif percentage_hash < (testing_percentage + validation_percentage):
    result = 'testing'
  else:
    result = 'training'
  return result

def is_audio_file(filename):
    return any(filename.endswith(extension) for extension in AUDIO_EXTENSIONS)


# def find_classes(dir):
#     classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
#     classes.sort()
    
#     return classes, class_to_idx


# def make_dataset(dir, class_to_idx):
#     spects = []
#     dir = os.path.expanduser(dir)
#     for target in sorted(os.listdir(dir)):
#         d = os.path.join(dir, target)
#         if not os.path.isdir(d):
#             continue

#         for root, _, fnames in sorted(os.walk(d)):
#             for fname in sorted(fnames):
#                 if is_audio_file(fname):
#                     path = os.path.join(root, fname)
#                     item = (path, class_to_idx[target])
#                     spects.append(item)
#     return spects
def do_time_shift(y,time_shift):
    # if time shift, get random shift value
    time_shift_amount = np.random.randint(-time_shift, time_shift)
    if time_shift_amount > 0: 
      time_shift_padding = [time_shift_amount, 0] 
      y = np.pad(y,time_shift_padding,'constant') #shift forward so left pad
      y = y[0:16000] #slice to first second
    else:
      time_shift_padding = [0, -time_shift_amount]
      y = np.pad(y,time_shift_padding,'constant') #shift backward so right pad
      y = y[-time_shift_amount:] #slice to last second
    return y

def wav_loader(path):
    y, sr = librosa.load(path, sr=None)
    return y,sr

def add_background_sample(y,sample):
    input_length = len(y)
    sample_start = random.randint(0, len(sample)-input_length)
    sample_sample = sample[sample_start:(sample_start + input_length)]
    return y + sample_sample

def spect_loader(y,sr, window_size, window_stride, window, normalize, max_len=101):
    

    n_fft = int(sr * window_size)
    win_length = n_fft
    hop_length = int(sr * window_stride)
    sample_length = len(y)
    
    # STFT
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length,
                     win_length=win_length, window=window)
    spect, phase = librosa.magphase(D)

    # S = log(S+1)
    spect = np.log1p(spect)

    # make all spects with the same dims
    # TODO: change that in the future
    if spect.shape[1] < max_len:
        pad = np.zeros((spect.shape[0], max_len - spect.shape[1]))
        spect = np.hstack((spect, pad))
    elif spect.shape[1] > max_len:
        spect = spect[:max_len, ]
    spect = np.resize(spect, (1, spect.shape[0], spect.shape[1]))
    spect = torch.FloatTensor(spect)

    # z-score normalization
    if normalize:
        mean = spect.mean()
        std = spect.std()
        if std != 0:
            spect.add_(-mean)
            spect.div_(std)   

    return spect

class GCommandLoader(data.Dataset):
    """A google command data set loader where the wavs are arranged in this way: ::
        [
            {'file': '../../gsk_train/left/3cdecb0b_nohash_0.wav', 'label': 'silence'}
            ...
        ]
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        window_size: window size for the stft, default value is .02
        window_stride: window stride for the stft, default value is .01
        window_type: typye of window to extract the stft, default value is 'hamming'
        normalize: boolean, whether or not to normalize the spect to have zero mean and one std
        max_len: the maximum length of frames to use
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        spects (list): List of (spects path, class_index) tuples
        STFT parameter: window_size, window_stride, window_type, normalize
    """

    def __init__(self, data_index, classes, class_to_idx, background_data=[],background_frequency=0,background_volume=0.3,transform=None, target_transform=None, window_size=.02,
                 window_stride=.01, time_shift=0, window_type='hamming', normalize=True, max_len=101):
        
        #self.data_dir = data_dir
        
        spects = data_index
        if len(spects) == 0:
            raise (RuntimeError("Found 0 sound files in subfolders of: " + root + "Supported audio file extensions are: " + ",".join(AUDIO_EXTENSIONS)))

        self.spects = spects
        self.background_data = background_data
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = spect_loader
        self.window_size = window_size
        self.window_stride = window_stride
        self.window_type = window_type
        self.normalize = normalize
        self.max_len = max_len
        self.time_shift = time_shift
        self.background_frequency = background_frequency
        self.background_volume = background_volume
        self.background_spects = self.load_background_data()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (spect, target) where target is class_index of the target class.
        """
        #print(index)
        path = self.spects[index]['file']
        label = self.spects[index]['label']
        target = self.class_to_idx[label]

        y, sr = librosa.load(path, sr=None)

        if label == 'silence':
            y = np.multiply(y,0)

        if self.time_shift > 0: 
            y = do_time_shift(y,self.time_shift)

        if random.random() < self.background_frequency:
            background_index = random.randint(0,len(self.background_spects)-1)
            background_spect = self.background_spects[background_index] 
            background_spect = np.multiply(background_spect,self.background_volume)
            y = add_background_sample(y,background_spect)

        spect = self.loader(y, sr, self.window_size, self.window_stride, self.window_type, self.normalize, max_len=self.max_len)
        

        return spect, target

    def __len__(self):
        return len(self.spects)

    def load_background_data(self):
      spects = []
      
      for path in self.background_data:
        y, sr = librosa.load(path, sr=None)
        spects.append(y) # should I care about sample rate? for now fixed at 16000/s

      return spects

    
class AudioProcessor(object):
  """Handles loading, partitioning, and preparing audio training data."""

  def __init__(self, data_url, data_dir, silence_percentage, unknown_percentage,
               wanted_words, validation_percentage, testing_percentage):
    self.data_dir = data_dir
    #self.maybe_download_and_extract_dataset(data_url, data_dir)
    self.prepare_data_index(silence_percentage, unknown_percentage,
                            wanted_words, validation_percentage,
                            testing_percentage)
    self.prepare_background_data()
    #self.prepare_processing_graph(model_settings)

  def maybe_download_and_extract_dataset(self, data_url, dest_directory):
    """Download and extract data set tar file.

    If the data set we're using doesn't already exist, this function
    downloads it from the TensorFlow.org website and unpacks it into a
    directory.
    If the data_url is none, don't download anything and expect the data
    directory to contain the correct files already.

    Args:
      data_url: Web location of the tar file containing the data set.
      dest_directory: File path to extract data to.
    """
    if not data_url:
      return
    if not os.path.exists(dest_directory):
      os.makedirs(dest_directory)
    filename = data_url.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):

      def _progress(count, block_size, total_size):
        sys.stdout.write(
            '\r>> Downloading %s %.1f%%' %
            (filename, float(count * block_size) / float(total_size) * 100.0))
        sys.stdout.flush()

      try:
        filepath, _ = urllib.request.urlretrieve(data_url, filepath, _progress)
      except:
        print('Failed to download URL: %s to folder: %s' % (data_url,filepath))
        print('Please make sure you have enough free space and'
                         ' an internet connection')
        raise
      print()
      statinfo = os.stat(filepath)
      print('Successfully downloaded %s (%d bytes)' % (filename,
                      statinfo.st_size))
    print ('unzipping %s' % filepath)
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)

  def prepare_background_data(self):
      """Searches a folder for background noise audio, and loads it into memory.

      It's expected that the background audio samples will be in a subdirectory
      named '_background_noise_' inside the 'data_dir' folder, as .wavs that match
      the sample rate of the training data, but can be much longer in duration.

      If the '_background_noise_' folder doesn't exist at all, this isn't an
      error, it's just taken to mean that no background noise augmentation should
      be used. If the folder does exist, but it's empty, that's treated as an
      error.

      Returns:
        List of raw PCM-encoded audio samples of background noise.

      Raises:
        Exception: If files aren't found in the folder.
      """
      self.background_data = []
      background_dir = os.path.join(self.data_dir, BACKGROUND_NOISE_DIR_NAME)
      if not os.path.exists(background_dir):
        return self.background_data
    
      search_path = os.path.join(self.data_dir, BACKGROUND_NOISE_DIR_NAME,
                                   '*.wav')
      for wav_path in gfile.Glob(search_path):
        self.background_data.append(wav_path)

      if not self.background_data:
        raise Exception('No background wav files were found in ' + search_path)

  def prepare_data_index(self, silence_percentage, unknown_percentage,
                         wanted_words, validation_percentage,
                         testing_percentage):
    """Prepares a list of the samples organized by set and label.

    The training loop needs a list of all the available data, organized by
    which partition it should belong to, and with ground truth labels attached.
    This function analyzes the folders below the `data_dir`, figures out the
    right
    labels for each file based on the name of the subdirectory it belongs to,
    and uses a satble hash to assign it to a data set partition.

    Args:
      silence_percentage: How much of the resulting data should be background.
      unknown_percentage: How much should be audio outside the wanted classes.
      wanted_words: Labels of the classes we want to be able to recognize.
      validation_percentage: How much of the data set to use for validation.
      testing_percentage: How much of the data set to use for testing.

    Returns:
      Dictionary containing a list of file information for each set partition,
      and a lookup map for each class to determine its numeric index.

    Raises:
      Exception: If expected files are not found.
    """
    # Make sure the shuffling and picking of unknowns is deterministic.
    random.seed(RANDOM_SEED)
    wanted_words_index = {}
    for index, wanted_word in enumerate(wanted_words):
      wanted_words_index[wanted_word] = index + 2
    self.data_index = {'validation': [], 'testing': [], 'training': []}
    unknown_index = {'validation': [], 'testing': [], 'training': []}
    all_words = {}
    # Look through all the subfolders to find audio samples
    search_path = os.path.join(self.data_dir, '*', '*.wav')
    print('# files in training: %d', len(gfile.Glob(search_path)))
    for wav_path in gfile.Glob(search_path):
      _, word = os.path.split(os.path.dirname(wav_path))
      word = word.lower()
      
      # Treat the '_background_noise_' folder as a special case, since we expect
      # it to contain long audio samples we mix in to improve training.
      if word == BACKGROUND_NOISE_DIR_NAME:
        continue
      all_words[word] = True
      set_index = which_set(wav_path, validation_percentage, testing_percentage)
      
      # If it's a known class, store its detail, otherwise add it to the list
      # we'll use to train the unknown label.
      if word in wanted_words_index:
        self.data_index[set_index].append({'label': word, 'file': wav_path})
      else:
        unknown_index[set_index].append({'label': word, 'file': wav_path})
    if not all_words:
      raise Exception('No .wavs found at ' + search_path)
    for index, wanted_word in enumerate(wanted_words):
      if wanted_word not in all_words:
        raise Exception('Expected to find ' + wanted_word +
                        ' in labels but only found ' +
                        ', '.join(all_words.keys()))
    # We need an arbitrary file to load as the input for the silence samples.
    # It's multiplied by zero later, so the content doesn't matter.
    silence_wav_path = self.data_index['training'][0]['file']
    for set_index in ['validation', 'testing', 'training']:
      set_size = len(self.data_index[set_index])
      print('size of %s: %d' % (set_index, set_size))
      silence_size = int(math.ceil(set_size * silence_percentage / 100))
      for _ in range(silence_size):
        self.data_index[set_index].append({
            'label': SILENCE_LABEL,
            'file': silence_wav_path
        })
      # Pick some unknowns to add to each partition of the data set.
      random.shuffle(unknown_index[set_index])
      unknown_size = int(math.ceil(set_size * unknown_percentage / 100))
      self.data_index[set_index].extend(unknown_index[set_index][:unknown_size])
    # Make sure the ordering is random.
    for set_index in ['validation', 'testing', 'training']:
      random.shuffle(self.data_index[set_index])
    # Prepare the rest of the result data structure.
    self.words_list = prepare_words_list(wanted_words)
    self.word_to_index = {}
    for word in all_words:
      if word in wanted_words_index:
        self.word_to_index[word] = wanted_words_index[word]
      else:
        self.word_to_index[word] = UNKNOWN_WORD_INDEX
    self.word_to_index[SILENCE_LABEL] = SILENCE_INDEX

  