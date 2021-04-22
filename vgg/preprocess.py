# load dogs vs cats dataset, reshape and save to a new file
from os import listdir
from numpy import asarray
from numpy import save
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
# define location of dataset
folder = './data'
photos, labels = list(), list()
# enumerate files in the directory
for file in listdir(folder):
	# determine class
	file = folder+'/'+file
	for j in listdir(file):
		output = 0.0
		if file=='tiger':
			output = 1.0
		# load image
		photo = load_img(file +'/'+ j, target_size=(200, 200))
		# convert to numpy array
		photo = img_to_array(photo)
		# store
		photos.append(photo)
		labels.append(output)
# convert to a numpy arrays
photos = asarray(photos)
labels = asarray(labels)
print(photos.shape, labels.shape)
# save the reshaped photos
save('jackal_vs_tiger_photos.npy', photos)
save('jackal_vs_tiger_labels.npy', labels)