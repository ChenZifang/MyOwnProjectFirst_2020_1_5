import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

#create a new layer in the network

def new_layer(inputs,in_size,out_size,activation_fuction=None):
    Weight = tf.Variable(tf.random.normal([in_size,out_size]))      #shape like conbine in_size and out_size
    biases = tf.Variable(tf.zeros([1,out_size])+0.1)                #all value in biases begain 0.1
    Wx_b = tf.matmul(inputs,Weight) + biases                        #the fuction x * weights + b
    if activation_fuction is None:
        output = Wx_b
    else:
        output = activation_fuction(Wx_b)                           #charge the fuction character ,if lines fuction should acticatication_fuction to change the shape
    return output                                                   #should return the value we already caculated
