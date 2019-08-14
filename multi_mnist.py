from PIL import Image, ImageDraw
import numpy as np
import tensorflow.keras
from tensorflow.keras.datasets import mnist
(x_train, y_train), _ = mnist.load_data()
size = x_train.shape

def get_data(num):
	
	out = np.zeros((num,5,5,15), dtype=np.float32)
	in_data = np.zeros((num,200,200), dtype=np.float32)
	div = 200/5

	for cnt in range(num):
		new_im = Image.new('L', (200,200))
		n = np.random.randint(2, 6) #randomly choosing how many images to combine
		pos = np.random.randint(0, size[0], n) #choosing n images from the dataset
		sample = np.array([x_train[i] for i in pos])
		y_train_sample = np.array([y_train[i] for i in pos])
		dim = np.random.randint(30, 60, n) #choosing dimension for each n image

		k = 0
		while (k < n):
			img = Image.fromarray(sample[k])
			img = img.resize((dim[k],dim[k]))
			i,j = np.random.randint(4,200-dim[k]-4),np.random.randint(4,200-dim[k]-4)
			s = np.sum(np.array(new_im)[i-4:i+dim[k]+4,j-4:j+dim[k]+4])
			if (s==0.):
				new_im.paste(img, (j,i))

				x_sample, y_sample = j+0.5*dim[k], i+0.5*dim[k]
				x_index, y_index = int(x_sample//div), int(y_sample//div)
				out[cnt,x_index,y_index,1] = (x_sample%div)/div
				out[cnt,x_index,y_index,2] = (y_sample%div)/div
				out[cnt,x_index,y_index,0] = 1.

				hw_sample = img.size
				out[cnt,x_index,y_index,3] = hw_sample[0]/200
				out[cnt,x_index,y_index,4] = hw_sample[1]/200

				cls_sample = y_train_sample[k]
				out[cnt, x_index, y_index, 5+cls_sample] = 1.
				k += 1
		in_data[cnt,:,:]=np.array(new_im)
		del new_im
	return in_data, out

def show(image, labels):
	pass	


		


