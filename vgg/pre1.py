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
dataset_home = './data/'
subdirs = ['train/', 'test/']
for subdir in subdirs:
	# create label subdirectories
	labeldirs = ['jackal/', 'tiger/']
	for labldir in labeldirs:
		newdir = dataset_home + subdir + labldir
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

test_sloths = jackal_images[:split_1]
train_sloths = jackal_images[split_1:]

test_snakes = tiger_images[:split_1]
train_snakes = tiger_images[split_1:]

print(len(train_sloths))
print(len(test_sloths))
print(train_sloths)

src_directory = './data/'
dst_dir = 'train/'
for j in range(len(train_sloths)):
	src = src_directory + train_sloths[j]
	dst = folder + dst_dir + 'jackal/' + train_sloths[j]
	copyfile(src, dst)

	src = src_directory + train_snakes[j]
	dst = folder + dst_dir + 'tiger/' + train_snakes[j]
	copyfile(src, dst)


dst_dir = 'test/'
for j in range(len(test_sloths)):
	src = src_directory + test_sloths[j]
	dst = folder + dst_dir + 'jackal/' + test_sloths[j]
	copyfile(src, dst)

	src = src_directory + test_snakes[j]
	dst = folder + dst_dir + 'tiger/' + test_snakes[j]
	copyfile(src, dst)