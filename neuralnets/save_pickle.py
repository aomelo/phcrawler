
# coding: utf-8

# Deep Learning
# =============
#
# Assignment 1
# ------------
#
# The objective of this assignment is to learn about simple data curation practices, and familiarize you with some of the data we'll be reusing later.
#
# This notebook uses the [notMNIST](http://yaroslavvb.blogspot.com/2011/09/notmnist-dataset.html) dataset to be used with python experiments. This dataset is designed to look like the classic [MNIST](http://yann.lecun.com/exdb/mnist/) dataset, while looking a little more like real data: it's a harder task, and the data is a lot less 'clean' than MNIST.

# In[15]:

# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import os
import sys
import tarfile
from IPython.display import display, Image
from scipy import ndimage
from six.moves import cPickle as pickle
import h5py

#from cPickle import pickle


# First, we'll download the dataset to our local machine. The data consists of characters rendered in a variety of fonts on a 28x28 image. The labels are limited to 'A' through 'J' (10 classes). The training set has about 500k and the testset 19000 labelled examples. Given these sizes, it should be possible to train models quickly on any machine.

# Extract the dataset from the compressed .tar.gz file.
# This should give you a set of directories, labelled A through J.

# In[16]:

#num_classes = 10
num_classes = 8
black_and_white = True
np.random.seed(133)

def maybe_extract(filename, force=False):
    root = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .tar.gz
    data_folders = [
        os.path.join(root, d) for d in sorted(os.listdir(root))
        if os.path.isdir(os.path.join(root, d))]

    if os.path.join(root, "full") in data_folders:
        data_folders.remove(os.path.join(root, "full"))
    if len(data_folders) != num_classes:
        raise Exception(
            'Expected %d folders, one per class. Found %d instead.' % (
                num_classes, len(data_folders)))
    print(data_folders)
    return data_folders



data_directory = "../data"

train_folders = maybe_extract(data_directory)


# ---
# Problem 1
# ---------
#
# Let's take a peek at some of the data to make sure it looks sensible. Each exemplar should be an image of a character A through J rendered in a different font. Display a sample of the images that we just downloaded. Hint: you can use the package IPython.display.
#
# ---

# Now let's load the data in a more manageable format. Since, depending on your computer setup you might not be able to fit it all in memory, we'll load each class into a separate dataset, store them on disk and curate them independently. Later we'll merge them into a single dataset of manageable size.
#
# We'll convert the entire dataset into a 3D array (image index, x, y) of floating point values, normalized to have approximately zero mean and standard deviation ~0.5 to make training easier down the road.
#
# A few images might not be readable, we'll just skip them.

# In[17]:

num_channels = 3
if black_and_white:
    num_channels = 1

#image_height = 28
#image_width = 28
image_height = 240
image_width = 424
pixel_depth = 255.0  # Number of levels per pixel.

def crop(image):
    diff_height = image.shape[0] - image_height
    diff_width =  image.shape[1] - image_width
    offset_x = int(diff_height/2)
    offset_y = int(diff_width/2)
    return image[offset_x:offset_x+image_height,offset_y:offset_y+image_width]

def load_letter(folder, min_num_images):
    """Load the data for a single letter label."""
    image_files = os.listdir(folder)
    dataset = np.ndarray(shape=(len(image_files), image_height, image_width),
                         dtype=np.float32)
    if not black_and_white:
        dataset = np.ndarray(shape=(len(image_files), image_height, image_width, num_channels),
                             dtype=np.float32)

    image_index = 0
    print(folder)
    for image in os.listdir(folder):
        image_file = os.path.join(folder, image)
        try:
            image_data = (ndimage.imread(image_file).astype(float) -
                          pixel_depth / 2) / pixel_depth

            if black_and_white:
                image_data = np.mean(image_data,axis=2)

            if image_data.shape != (image_height, image_width):
                #raise Exception('Unexpected image shape: %s' % str(image_data.shape))
                image_data = crop(image_data)
            dataset[image_index, :, :] = image_data
            image_index += 1
        except IOError as e:
            print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')

    num_images = image_index
    dataset = dataset[0:num_images, :, :]
    if num_images < min_num_images:
        raise Exception('Many fewer images than expected: %d < %d' %
                        (num_images, min_num_images))

    print('Full dataset tensor:', dataset.shape)
    print('Mean:', np.mean(dataset))
    print('Standard deviation:', np.std(dataset))
    return dataset

def maybe_hfs5(data_folders, min_num_images_per_class, force=False):
    dataset_names = []
    for folder in data_folders:
        set_filename = folder + '.h5'
        dataset_names.append(set_filename)
        if os.path.exists(set_filename) and not force:
            # You may override by setting force=True.
            print('%s already present - Skipping hdf5.' % set_filename)
        else:
            print('Saving to hdf5 %s.' % set_filename)
            dataset = load_letter(folder, min_num_images_per_class)
            try:
                with h5py.File(set_filename, "w") as hf:
                    hf.create_dataset("dataset",data=dataset)
            except Exception as e:
                print('Unable to save data to', set_filename, ':', e)

    return dataset_names


datasets = maybe_hfs5(train_folders, 0)


# ---
# Problem 2
# ---------
#
# Let's verify that the data still looks good. Displaying a sample of the labels and images from the ndarray. Hint: you can use matplotlib.pyplot.
#
# ---

# ---
# Problem 3
# ---------
# Another check: we expect the data to be balanced across classes. Verify that.
#
# ---

# Merge and prune the training data as needed. Depending on your computer setup, you might not be able to fit it all in memory, and you can tune `train_size` as needed. The labels will be stored into a separate array of integers 0 through 9.
#
# Also create a validation dataset for hyperparameter tuning.

# In[7]:

def make_arrays(nb_rows, img_height, img_width):
    dataset, labels = None, None
    if nb_rows:
        if black_and_white:
            dataset = np.ndarray((nb_rows, img_height, img_width), dtype=np.float32)
        else:
            dataset = np.ndarray((nb_rows, img_height, img_width, num_channels), dtype=np.float32)
        labels = np.ndarray(nb_rows, dtype=np.int32)
    return dataset, labels


def load_datasets(hdf5_files, train_size, valid_size=0, test_size=0):
    num_classes = len(hdf5_files)
    valid_dataset, valid_labels = make_arrays(valid_size, image_height, image_width)
    test_dataset , test_labels  = make_arrays(test_size, image_height, image_width)
    train_dataset, train_labels = make_arrays(train_size, image_height, image_width)

    dataset_size = train_size+test_size+valid_size
    dataset, labels = make_arrays(dataset_size, image_height, image_width)

    offset = 0
    for label, hdf5_file in enumerate(hdf5_files):
        print(hdf5_file)
        try:
            with h5py.File(hdf5_file,'r') as hf:
                letter_set = hf.get('dataset')
                num_class_instances = letter_set.shape[0]
                dataset[offset:offset+num_class_instances,:,:] = letter_set
                labels[offset:offset+num_class_instances] = label
                offset = offset + num_class_instances
        except Exception as e:
            print('Unable to process data from', hdf5_file, ':', e)
            raise

    indices = np.zeros((dataset_size))
    indices[0:test_size] = np.ones((test_size))
    indices[test_size:test_size+valid_size] = np.full((valid_size),2)

    np.random.shuffle(indices)

    train_dataset = dataset[np.where(indices==0)[0],:,:]
    test_dataset  = dataset[np.where(indices==1)[0],:,:]
    valid_dataset = dataset[np.where(indices==2)[0],:,:]

    train_labels  = labels[np.where(indices==0)[0]]
    test_labels  = labels[np.where(indices==1)[0]]
    valid_labels = labels[np.where(indices==2)[0]]


    return test_dataset, test_labels, valid_dataset, valid_labels, train_dataset, train_labels


train_size = 3902
valid_size = 500
test_size = 500
#train_size = 1744
#valid_size = 100
#test_size = 100


#valid_dataset, valid_labels, train_dataset, train_labels = merge_datasets(
#  train_datasets, train_size, valid_size)
#_, _, test_dataset, test_labels = merge_datasets(test_datasets, test_size)

test_dataset, test_labels, \
valid_dataset, valid_labels, \
train_dataset, train_labels = load_datasets(datasets,train_size,valid_size,test_size)

print('Training:', train_dataset.shape, train_labels.shape)
print('Validation:', valid_dataset.shape, valid_labels.shape)
print('Testing:', test_dataset.shape, test_labels.shape)


# Next, we'll randomize the data. It's important to have the labels well shuffled for the training and test distributions to match.

# In[ ]:

def randomize(dataset, labels):
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation,:,:]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels
train_dataset, train_labels = randomize(train_dataset, train_labels)
test_dataset, test_labels = randomize(test_dataset, test_labels)
valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)


# ---
# Problem 4
# ---------
# Convince yourself that the data is still good after shuffling!
#
# ---

# Finally, let's save the data for later reuse:

# In[ ]:

pickle_file = 'ph.h5'
if black_and_white:
    pickle_file = pickle_file.replace(".h5","-bw.h5")

try:
    with h5py.File(pickle_file.replace(".pickle",".h5"), "w") as hf:
        hf.create_dataset("train_dataset",data=train_dataset)
        hf.create_dataset("train_labels",data=train_labels)
        hf.create_dataset("valid_dataset",data=valid_dataset)
        hf.create_dataset("valid_labels",data=valid_labels)
        hf.create_dataset("test_dataset",data=test_dataset)
        hf.create_dataset("test_labels",data=test_labels)
except Exception as e:
    print('Unable to save data to', pickle_file, ':', e)
    raise


# In[ ]:

statinfo = os.stat(pickle_file)
print('Compressed pickle size:', statinfo.st_size)


# ---
# Problem 5
# ---------
#
# By construction, this dataset might contain a lot of overlapping samples, including training data that's also contained in the validation and test set! Overlap between training and test can skew the results if you expect to use your model in an environment where there is never an overlap, but are actually ok if you expect to see training samples recur when you use it.
# Measure how much overlap there is between training, validation and test samples.
#
# Optional questions:
# - What about near duplicates between datasets? (images that are almost identical)
# - Create a sanitized validation and test set, and compare your accuracy on those in subsequent assignments.
# ---

# ---
# Problem 6
# ---------
#
# Let's get an idea of what an off-the-shelf classifier can give you on this data. It's always good to check that there is something to learn, and that it's a problem that is not so trivial that a canned solution solves it.
#
# Train a simple model on this data using 50, 100, 1000 and 5000 training samples. Hint: you can use the LogisticRegression model from sklearn.linear_model.
#
# Optional question: train an off-the-shelf model on all the data!
#
# ---

# In[ ]:




# In[ ]:



