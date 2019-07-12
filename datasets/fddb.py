import os
import torch

from torch.utils.data import Dataset, DataLoader
from PIL import Image

class FaceDataset(Dataset):
	"""FDDB (http://vis-www.cs.umass.edu/fddb/)"""

	def __init__(self, annotations, root_dir, transform=None):
		"""
		Args:
			annotations (string): Path to the text file with annotations.
			root_dir (string): Directory with all the images.
			transform (callable, optional): Optional transform to be applied
				on a sample.
		"""
		self.root_dir = root_dir
		self.transform = transform
		self.annotations = []
		
		file = open(annotations, 'r')
		
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
		
			
			
	def __len__(self):
		return len(self.annotations)

	def __getitem__(self, idx):
		img_name, x, y = self.annotations[idx]
		
		img = load_image(os.path.join(self.root_dir, img_name))

		if self.transform is not None:
			img = self.transform(img)

		return img, torch.Tensor([x, y])
		
		
def load_image(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')