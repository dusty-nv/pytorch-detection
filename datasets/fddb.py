import os
import math
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
				
					if num_faces != 1:
						continue

					# <major_axis_radius minor_axis_radius angle center_x center_y 1>
					radius_major = float(face_str[0])
					radius_minor = float(face_str[1])

					angle = float(face_str[2])

					center_x = float(face_str[3])
					center_y = float(face_str[4])

					# compute bounding-box of ellipse (https://stackoverflow.com/a/14163413)
					ux = radius_major * math.cos(angle);
					uy = radius_major * math.sin(angle);
					vx = radius_minor * math.cos(angle + math.pi * 0.5);
					vy = radius_minor * math.sin(angle + math.pi * 0.5);

					bbox_halfwidth = math.sqrt(ux*ux + vx*vx);
					bbox_halfheight = math.sqrt(uy*uy + vy*vy); 

					bbox_left = center_x - bbox_halfwidth
					bbox_right = center_x + bbox_halfwidth

					bbox_top = center_y - bbox_halfheight
					bbox_bottom = center_y + bbox_halfheight

					#print('{:s}  ({:f}, {:f}) ({:f}, {:f})'.format(img_name, bbox_left, bbox_top, bbox_right, bbox_bottom))
					self.annotations.append((img_name, bbox_left, bbox_top, bbox_right, bbox_bottom))
			
			file.close()
		
	def output_dims(self):
		return 4	# bbox left, top, right, bottom
			
	def __len__(self):
		return len(self.annotations)

	def __getitem__(self, idx):
		img_name, left, top, right, bottom = self.annotations[idx]
		
		img = load_image(os.path.join(self.root_dir, img_name))

		org_width = float(img.width)
		org_height = float(img.height)

		if self.transform is not None:
			img = self.transform(img)
			
		width = float(img.size()[2])
		height = float(img.size()[1])

		left = normalize(left, org_width, width)
		right = normalize(right, org_width, width)
		top = normalize(top, org_height, height)
		bottom = normalize(bottom, org_height, height)

		#print('original size:  {:d}x{:d}   transformed size:  {:d}x{:d}   scale factor:  {:f}x{:f}'.format(org_width, org_height, width, height, scale_width, scale_height))
		return img, torch.Tensor([left, top, right, bottom])


def normalize(coord, original_dim, rescaled_dim):
	scale = rescaled_dim / original_dim
	coord = coord * scale
	return 2.0 * (coord / rescaled_dim - 0.5)	
		

def load_image(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
