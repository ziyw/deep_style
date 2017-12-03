import caffe
import cv2
import numpy as np
from skimage import transform


class DeepStyle():

    def __init__(self, style_net_file, alpha=1, beta=1):

        self.image_size = 224
        self.alpha = alpha
        self.beta = beta 

        # self.net = caffe.Net(style_net_file, caffe.TEST)
        pass 


    def transfer_style(self, style_image_file, content_image_file):

        self.style_image = self.read_image(style_image_file)
        self.content_image = self.read_image(content_image_file)
        
        self.x = self.generate_white_noise_image()

        self.train_network()

        print 'Done training'

        pass



    def read_image(self, image_file):
    	# TODO, warning : what should be the size of image 
        img = cv2.imread(image_file, 1)
        resized_img = transform.resize(img, (self.image_size, self.image_size), mode='constant') 
        resized_img = resized_img.shape(3,self.image_size, self.image_size)
        return resized_img[np.newaxis, :, :, :]

    def generate_white_noise_image(self):
    	gaussian_image = np.random.normal((self,image_size, self.image_size, 3))
    	return gaussian_image[np.newaxis, :, :, :]

    

    def train_network(self):

        pass


    def visulize(self, layer_name):

    	pass



def main():

	pass 

if __name__ == '__main__':
    main()
