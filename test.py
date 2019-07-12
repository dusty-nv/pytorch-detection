import os
import torch
import torchvision

from datasets import FaceDataset

face_dataset = FaceDataset(annotations='../fddb-annotations.txt',
                           root_dir='../originalPics')

print('num_faces = {:d}'.format(len(face_dataset)))

for i in range(len(face_dataset)):
	img, coord = face_dataset[i]
	print('image {:d}: {:d}x{:d}'.format(i, img.width, img.height))
	print(coord)