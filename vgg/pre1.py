from os import listdir
from numpy import asarray
from numpy import save

from os import makedirs
from os import listdir
from shutil import copyfile
from random import seed
from random import random
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

# create directories
dataDirectory = './data/'
subdirs = ['train/', 'test/']
for subdir in subdirs:
	# create label subdirectories
	labeldirs = ['jackal/', 'tiger/']
	for labldir in labeldirs:
		newdir = dataDirectory + subdir + labldir
		makedirs(newdir, exist_ok=True)

import random
jackal_images = []
tiger_images = []

folder = './data/'
for file in listdir(folder):
	if file.startswith('jackal'):
		jackal_images.append(file)
	elif file.startswith('tiger'):
		tiger_images.append(file)

random.seed(42)
random.shuffle(jackal_images) # shuffles the ordering of filenames (deterministic given the chosen seed)
random.shuffle(tiger_images) # shuffles the ordering of filenames (deterministic given the chosen seed)

split_1 = int(0.25 * len(jackal_images))

test_jackal = jackal_images[:split_1]
train_jackal = jackal_images[split_1:]

test_tiger = tiger_images[:split_1]
train_tiger = tiger_images[split_1:]

print(len(train_jackal))
print(len(test_jackal))
print(train_jackal)

src_directory = './data/'
dst_dir = 'train/'
for j in range(len(train_jackal)):
	src = src_directory + train_jackal[j]
	dst = folder + dst_dir + 'jackal/' + train_jackal[j]
	copyfile(src, dst)

	src = src_directory + train_tiger[j]
	dst = folder + dst_dir + 'tiger/' + train_tiger[j]
	copyfile(src, dst)


dst_dir = 'test/'
for j in range(len(test_jackal)):
	src = src_directory + test_jackal[j]
	dst = folder + dst_dir + 'jackal/' + test_jackal[j]
	copyfile(src, dst)

	src = src_directory + test_tiger[j]
	dst = folder + dst_dir + 'tiger/' + test_tiger[j]
	copyfile(src, dst)