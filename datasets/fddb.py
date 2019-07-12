import os
import torch

from torch.utils.data import Dataset, DataLoader
from PIL import Image

class FaceDataset(Dataset):
	"""FDDB (http://vis-www.cs.umass.edu/fddb/)"""

	def __init__(self, root_dir, fold_dir, fold_range, transform=None):
		"""
		Args:
			root_dir (string): Directory with all the images.
			fold_dir (string): Directory with the annotation folds.
			fold_range (range): Range of annotation folds to load.
			transform (callable, optional): Optional transform to be applied
				on a sample.
		"""
		self.root_dir = root_dir
		self.transform = transform
		self.annotations = []
		
		for f in fold_range:
			file = open(os.path.join(fold_dir, "FDDB-fold-{:02d}-ellipseList.txt".format(f)), 'r')
		
			while True:
				img_name = file.readline()
			
				if not img_name:
					break
				
				img_name = img_name.rstrip() + ".jpg"
				#print('img_name = ' + img_name)
			
				num_faces = int(file.readline())
				#print('num_faces = {:d}'.format(num_faces))
			
				for n in range(num_faces):
					face_str = file.readline().rstrip().split(' ')
					#print(face_str)
				
					if num_faces == 1:
						self.annotations.append((img_name, float(face_str[3]), float(face_str[4])))
			
			file.close()
		
	def output_dims(self):
		return 2	# x and y
			
	def __len__(self):
		return len(self.annotations)

	def __getitem__(self, idx):
		img_name, x, y = self.annotations[idx]
		
		img = load_image(os.path.join(self.root_dir, img_name))

		width = img.width
		height = img.height

		if self.transform is not None:
			img = self.transform(img)

		x = 2.0 * (x / width - 0.5) # -1 left, +1 right
		y = 2.0 * (y / height - 0.5) # -1 top, +1 bottom

		return img, torch.Tensor([x, y])

		
		
def load_image(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
