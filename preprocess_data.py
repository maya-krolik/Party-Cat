import random
import cv2, os
import pandas as pd
import numpy as np

image_size = 258
for i in range(0,6):
  dirname = "CAT_0%d" %i
base_path = "dataset/%s" % dirname # paths of all raw data values
file_list = sorted(os.listdir(base_path))
random.shuffle(file_list) # randomize

# define dataset
dataset = {
  "imgs": [],
  "lmks": [],
}

#-------------------------------------------------------------------------------
def resize_image(im):
  old_size = im.shape[:2] # old_size is in (height, width) format
  ratio = float(image_size) / max(old_size)
  new_size = tuple([int(x*ratio) for x in old_size])

  im = cv2.resize(im, (new_size[1], new_size[0]))
  delta_w = image_size - new_size[1]
  delta_h = image_size - new_size[0]
  top, bottom = delta_h // 2, delta_h - (delta_h // 2)
  left, right = delta_w // 2, delta_w - (delta_w // 2)
  # resize image according to calculations
  new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
      value=[0, 0, 0])
  return new_im, ratio, top, left

#-------------------------------------------------------------------------------
for f in file_list:
  if '.cat' not in f: # isolate images
    continue

  # read landmarks from .cat files
  landmarks = pd.read_csv(os.path.join(base_path, f), sep=' ', header=None)

  # load image
  img_filename, ext = os.path.splitext(f)
  img = cv2.imread(os.path.join(base_path, img_filename))

  # resize image and relocate landmarks based on resize ratio
  img, ratio, top, left = resize_image(img)
  new_landmarks = (landmarks * ratio)

  # create database
  dataset['imgs'].append(img)
  dataset['lmks'].append(new_landmarks.stack().values)

# save database
np.save('dataset/lmks_%s.npy' % dirname, np.array(dataset))