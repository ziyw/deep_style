import caffe
import cv2
import numpy as np
from skimage import transform
import matplotlib.pyplot as plt 


class DeepStyle():

    def __init__(self, model_prototxt, alpha=1, beta=1):

        self.model_prototxt = model_prototxt 

        self.net = caffe.Net(model_prototxt, caffe.TEST)

        self.alpha = alpha
        self.beta = beta 

        self.image_size = 224
        
    def transfer_style(self, style_image_file, content_image_file):

        self.style_image = self.read_image(style_image_file)
        self.content_image = self.read_image(content_image_file)
        
        self.output_image = self.generate_white_noise_image()

        self.train()
        print 'Done training'


    def read_image(self, image_file):
        img = cv2.imread(image_file, 1)
        resized_img = transform.resize(img, (self.image_size, self.image_size), mode='constant') 

        resized_img = resized_img.reshape(3,self.image_size, self.image_size)
      
        return resized_img[np.newaxis, :, :, :]

    def generate_white_noise_image(self):
    	gaussian_image = np.random.normal(0,1, (3,self.image_size, self.image_size))
    	return gaussian_image[np.newaxis, :, :, :]

    def train(self):

        self.get_content_layers()
        self.get_style_layers()

        self.net.blobs['data'].data = self.output_image

        for iteration in range(number_of_iterations):
                
            self.net.forward()

            # net.blobs["loss"].data = 
            # Clear the diffs, then run the backward step
            for name, l in zip(self.net._layer_names, self.net.layers):
                for b in l.blobs:
                    pass 
                    # b.diff[...] = content_gradient[name] + style_gradient[name] 

            net.backward()

            for l in self.net.layers:
                for b in l.blobs:
                    pass 
                    # b.data[...] -= learning_rate * b.diff

            learning_rate = base_lr * math.pow(1 + gamma * iter_num, - power)
                
            
        pass


    def get_content_layers(self):

        self.net.blobs['data'].reshape(*(self.content_image.shape))
        self.net.blobs['data'].data[...] = self.content_image

        self.net.forward()
        
        P = {}
        layers = [n for n in self.net._layer_names]
        for l in layers:
            if l == 'input' or l[:4] == 'relu':
                continue
            P[l] = self.net.blobs[l].data

        self.content_layers = P

    def get_style_layers(self):

        A = {}

        self.net.blobs['data'].reshape(*(self.style_image.shape))
        self.net.blobs['data'].data[...] = self.style_image

        self.net.forward()

        layers = [n for n in self.net._layer_names]

        for l in layers:
            if l == 'input' or l[:4] == 'relu':
                continue
            A[l] = self.net.blobs[l].data

        self.style_image = A

    def visulize(self, layer_name):

    	pass



def main():
    model = DeepStyle('style_model.prototxt')
    
    content_image_file = '../input_image/cat.jpg'
    style_image_file = '../input_image/social_network.jpg'

    model.transfer_style(content_image_file, style_image_file)


if __name__ == '__main__':
    main()
