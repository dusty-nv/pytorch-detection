import os
import torch
import torchvision
import torchvision.transforms as transforms

from datasets import FaceDataset

face_dataset = FaceDataset(root_dir='../originalPics',
					  fold_dir='../FDDB-folds',
					  fold_range=range(1,11),
                           transform=transforms.Compose([
						transforms.Resize((224,224)),
						#transforms.RandomResizedCrop(args.resolution),
						transforms.RandomHorizontalFlip(),
						transforms.ToTensor()
						]))

print('num_faces = {:d}'.format(len(face_dataset)))

for i in range(len(face_dataset)):
	img, coord = face_dataset[i]
	#print('image {:d}: {:d}x{:d}'.format(i, img.width, img.height))
	print(img.size())
	print(coord)
