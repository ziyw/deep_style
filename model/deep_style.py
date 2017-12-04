import caffe
import cv2
import numpy as np
from skimage import transform


class DeepStyle():

    def __init__(self, model_prototxt, alpha=1, beta=1):

        self.net = caffe.Net(model_prototxt, caffe.TEST)

        self.alpha = alpha
        self.beta = beta 

        self.image_size = 224
        
    def transfer_style(self, style_image_file, content_image_file):

        self.style_image = self.read_image(style_image_file)
        self.content_image = self.read_image(content_image_file)
        
        self.x = self.generate_white_noise_image()

        self.train()



        # print self.net.blobs.items

        # solver = caffe.SGDSolver('solver.prototxt')

        # solver.solve()


        # set input data 
        # self.net.blobs['data'] = input images 

        # forward 
        # self.net.forward()




        # backward 
        # calculate gradient 
        # self.net.backward() 

        # Solver?? Necessary??

        self.train()

        # WARNING, save the result as output image 
        print 'Done training'


    def read_image(self, image_file):
        img = cv2.imread(image_file, 1)
        resized_img = transform.resize(img, (self.image_size, self.image_size), mode='constant') 

        resized_img = resized_img.reshape(3,self.image_size, self.image_size)
      
        return resized_img[np.newaxis, :, :, :]

    def generate_white_noise_image(self):
    	gaussian_image = np.random.normal(0,1, (3,self.image_size, self.image_size))
    	return gaussian_image[np.newaxis, :, :, :]


    def set_data_layer(self):
        self.net.blobs['data'] = self.x

    def train(self):


        # use transformer to calculate content_image  and style_image value in each layer 
        # They will not be changed so they only need to be update once 

        # after change those two
        # the loss of gradient of input x into each layer is different everytime 
        # gradient 




        num_epochs = 2 # How many times we are going to run through the database
        iter_num = 0 # Current iteration number

        # Training and testing examples
        db_path = "examples/mnist/mnist_train_lmdb"
        db_path_test = "examples/mnist/mnist_test_lmdb"

        # Learning rate. We are using the lr_policy "inv", here, with no momentum
        base_lr = 0.01
        # Parameters with which to update the learning rate
        gamma = 1e-4
        power = 0.75

        for epoch in range(num_epochs):
            print("Starting epoch {}".format(epoch))
            # At each epoch, iterate over the whole database
            input_shape = net.blobs["data"].data.shape
            for batch in batch_generator(input_shape, db_path):
                iter_num += 1
                
                # Run the forward step
                net.blobs["data"].data[...] = batch
                net.forward()
                
                # Clear the diffs, then run the backward step
                for name, l in zip(net._layer_names, net.layers):
                    for b in l.blobs:
                        b.diff[...] = net.blob_loss_weights[name]
                net.backward()
                
                # Update the learning rate, with the "inv" lr_policy
                learning_rate = base_lr * math.pow(1 + gamma * iter_num, - power)
                
                # Apply the diffs, with the learning rate
                for l in net.layers:
                    for b in l.blobs:
                        b.data[...] -= learning_rate * b.diff
                
                # Display the loss every 50 iterations
                if iter_num % 50 == 0:
                    print("Iter {}: loss={}".format(iter_num, net.blobs["loss"].data))
                    
                # Test the network every 200 iterations
                if iter_num % 200 == 0:
                    print("Testing network: accuracy={}, loss={}".format(*test_network(test_net, db_path_test)))

        print("Training finished after {} iterations".format(iter_num))
        print("Final performance: accuracy={}, loss={}".format(*test_network(test_net, db_path_test)))
        # Save the weights
        net.save("examples/mnist/lenet_iter_{}.caffemodel".format(iter_num))



        print self.net.forward()

        # keep track of content in each layer 


        





        pass


    def visulize(self, layer_name):

    	pass



def main():
    model = DeepStyle('style_model.prototxt')
    
    content_image_file = '../input_image/cat.jpg'
    style_image_file = '../input_image/social_network.jpg'

    model.transfer_style(content_image_file, style_image_file)


if __name__ == '__main__':
    main()
