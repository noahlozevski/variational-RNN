# import pandas as pd
import numpy as np
import os
from collections import defaultdict
from pydash import _

DATASET_PATH = os.getcwd().split('/variational_lstm')[0] + '/datasets/Task2'
ENGLISH = [2, 4, 6, 8, 10, 12, 13, 15, 18, 20, 22, 24, 25, 28, 30, 32, 33, 34, 35, 40]
# genuine signatures are indexed 1-20, forged are 21-40
langs = ['chinese', 'english']
processed = []
for i in range(1,41):
    values = np.array([])
    # _.for_each(list(range(1,41)), lambda x: values.append(np.loadtxt(f'{DATASET_PATH}/U{i}S{x}.TXT',skiprows=1)))
    # values = _.reduce(list(range(1,41)), lambda total, x: np.append(values, np.loadtxt(f'{DATASET_PATH}/U{i}S{x}.TXT',skiprows=1)))
    # for x in range(1,41):
    #   np.append(values, )
    # [np.append(values, np.loadtxt(f'{DATASET_PATH}/U{i}S{x}.TXT',skiprows=1) for x in range(1,41)]
    processed.append({ 'user': i,
                       'values': [np.loadtxt(f'{DATASET_PATH}/U{i}S{x}.TXT', skiprows=1) for x in range(1,41)],
                      #  'values': [np.loadtxt('{}/U{}S{}.TXT'.format(DATASET_PATH,i,x), skiprows=1) for x in range(1,41)],
                       'language': 'english' if i in ENGLISH else 'chinese' })

min_length = defaultdict(lambda:float('inf'))
max_length = defaultdict(lambda:float('-inf'))
for lang in langs:
  for val in _.filter(processed, lambda x: x['language'] == lang):
    curr_min = min(len(x) for x in val['values'])
    curr_max = max(len(x) for x in val['values'])
    if curr_min < min_length[lang]:
      min_length[lang] = curr_min
    if curr_max > min_length[lang]:
      max_length[lang] = curr_max
    
print(f'{len(processed)} signatures loaded\n\n')
print(f'English min length: {min_length["english"]}\nChinese min length: {min_length["chinese"]}\n\n')
print(f'English max length: {max_length["english"]}\nChinese max length: {max_length["chinese"]}')

# def get_english():
#   return _.filter(dataset, lambda x: x['language'] == 'english')

# def get_chinese():
#   return _.filter(dataset, lambda x: x['language'] == 'chinese')
def get_label(index):
  return index <= 20

def trim(arr,length):
  arr = arr[:length,:]
  return arr

def window(a, w = 4, o = 2, copy = False):
    sh = (a.size - w + 1, w)
    st = a.strides * 2
    view = np.lib.stride_tricks.as_strided(a, strides = st, shape = sh)[0::o]
    if copy:
        return view.copy()
    else:
        return view


def get_dataset(overlap=5, window_size=10, time_steps=20, language=None, max_len=80):
  if language:
    dataset = _.filter(processed, lambda x: x['language'] == language)
  else:
    dataset = processed
  num_features = 7 # number of features on each time step
  # x,y,meta = [],[],[]
  genuine = forged = None
  if not max_len:
    pass
  else:
    for val in dataset:
      update = False
      g = val['values'][:20]
      f = val['values'][20:]
      
      g = np.array(_.map(_.filter(g, lambda i: len(i) >= max_len), lambda x: window(trim(x,max_len).reshape(-1),window_size*num_features,overlap*num_features,True)))
      f = np.array(_.map(_.filter(f, lambda i: len(i) >= max_len), lambda x: window(trim(x,max_len).reshape(-1),window_size*num_features,overlap*num_features,True)))
      if genuine is None and g.size > 0:
        genuine = g
        update = True
      if forged is None and f.size > 0:
        forged = f
        update = True
        continue
      if (g.size > 0 or f.size > 0) and not update:
        genuine = genuine if g.size == 0 else np.append(genuine,g,axis=0)
        forged = forged if f.size == 0 else np.append(forged,f,axis=0)
      
    print(genuine.shape) # (num_samples, num_time_steps, window_size*num_features)
    print(f'Options\nWindow size --> {window_size}\nOverlap (in samples) --> {overlap}\nMax length {max_len}\n{"— "*20}')
    print(f'Number of time steps --> {genuine.shape[1]}\nNumber of features at each step --> {genuine.shape[2]}')
    print(f'Number of genuine samples --> {genuine.shape[0]}')
    print(f'Number of forged samples --> {forged.shape[0]}')
    x = { "genuine": genuine,
          'forged': forged }
    return x
      # y = np.array([i < 20 for i in range(40)])

      # each value is a 2d vector with shape (num_time_steps, window_size*num_features)
      
    # return dataset
      # for i,_x in enumerate(val['values']):
      #   x.append(_x[:max_len,:])
      #   if
          

        
        
    # return _.map(dataset, lambda x: )

