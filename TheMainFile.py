import TheNetWorkFuction as TNWF
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import matplotlib.pyplot as plt

#create datas
x_data = np.linspace(-1,1,500)[:,np.newaxis]   #create x and change the shape into matrix\

y_data = np.square(x_data) + 0.5

#middle datas
xs = tf.placeholder(tf.float32,[None,1])
noise = np.random.normal(0,0.05,x_data.shape)
ys = tf.placeholder(tf.float32,[None,1])

#create three parts one input ,ten hide_data input and ont final output,all should two network
#create a layer and give the activation_fuction as input_layer
Input_layer = TNWF.new_layer(xs,1,10,activation_fuction=tf.nn.relu)

#create a hide layer
Hide_layer = TNWF.new_layer(Input_layer,10,1,activation_fuction=None)

#save the loss and caculate the loss
loss = tf.reduce_mean(tf.reduce_sum(tf.square(Hide_layer - ys),reduction_indices=[1]))

Train_set = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

Init_variables = tf.initialize_all_variables()


#make resoult visable
#create a empty plot
Figure = plt.figure()

#fuction show Fuction_show
Fuction_show = plt.subplot(1,1,1)

Fuction_show.scatter(x_data,y_data)

#keep running and show
plt.ion()
plt.show()


with tf.Session() as sess:
    sess.run(Init_variables)
    for i in range(1000):
        sess.run(Train_set,feed_dict={xs:x_data,ys:y_data})
        if i % 50 == 0:
            try:
                # remove the first line
                Fuction_show.lines.remove(lines[0])
            except Exception:
                pass
            #print(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))
            Prediction_value = sess.run(Hide_layer,feed_dict={xs:x_data,ys:y_data})
            lines = Fuction_show.plot(x_data,Prediction_value,"r-",lw=5)
            plt.pause(0.1)


plt.pause(0)