import tensorflow.compat.v1 as  tf
tf.disable_v2_behavior()
import numpy as np
import matplotlib.pyplot as plt

#gloable seed
tf.set_random_seed(1)
np.random.seed(1)


#Gloabale_paraments
#Inmput_samples
I_samples = 20

#Hidden_layer samples
H_samples = 300

#Learning rata
LR = 0.01
#creates datas --- datas use np.newaxis change datas into type of tensor
#training-set datas
Train_x = np.linspace(-1,1,I_samples)[:,np.newaxis]
Train_y = Train_x + 0.3 * np.random.rand(I_samples)[:,np.newaxis]

#tests datas
Test_x = Train_x.copy()
Test_y = Test_x + 0.3 * np.random.rand(I_samples)[:,np.newaxis]

#show datas as plot
fig1 = plt.figure()
plt.scatter(Test_x,Train_y,c='magenta',s=50,alpha=0.5,label='Train')
plt.scatter(Test_x,Test_y,c="b",s=50,alpha=0.5,label='Text')
plt.show()

#Input Datas to prediction
tf_x = tf.placeholder(tf.float32,[None,1])
tf_y = tf.placeholder(tf.float32,[None,1])
tf_is_training = tf.placeholder(tf.bool, None)  # to control dropout when training and testing

#overflow net
o1 = tf.layers.dense(tf_x,H_samples,tf.nn.relu)
o2 = tf.layers.dense(o1,H_samples,tf.nn.relu)
O_out = tf.layers.dense(o2,1)
O_loss = tf.losses.mean_squared_error(tf_y,O_out)
O_train = tf.train.AdamOptimizer(LR).minimize(O_loss)

# dropout net 4 net layers to make more close need lineß
d1 = tf.layers.dense(tf_x, H_samples, tf.nn.relu)
d1 = tf.layers.dropout(d1, rate=0.5, training=tf_is_training)   # drop out 50% of inputs
d2 = tf.layers.dense(d1, H_samples, tf.nn.relu)
d2 = tf.layers.dropout(d2, rate=0.5, training=tf_is_training)   # drop out 50% of inputs
d_out = tf.layers.dense(d2, 1)
d_loss = tf.losses.mean_squared_error(tf_y, d_out)
d_train = tf.train.AdamOptimizer(LR).minimize(d_loss)

#session run
sess = tf.Session()
sess.run(tf.global_variables_initializer())

#make plt continue run
plt.ion()

for t in range(500):
    sess.run([O_train, d_train], {tf_x: Train_x, tf_y: Train_y, tf_is_training: True})

    if t % 10 == 0:
        #show plot
        plt.cla()
        o_loss_, d_loss_, o_out_, d_out_ = sess.run(
            [O_loss, d_loss, O_out, d_out], {tf_x: Test_x, tf_y: Train_y, tf_is_training: False}
            # test, set is_training=False
        )
        plt.scatter(Train_x,Train_y,c='magenta',s=50,alpha=0.3,label='train')
        plt.scatter(Test_x,Test_y,c='cyan',s=50,alpha=0.3,label='test')
        plt.plot(Test_x, o_out_, 'r-', lw=3, label='overfitting')
        plt.plot(Test_x, d_out_, 'b--', lw=3, label='dropout(50%)')
        plt.text(0, -1.2, 'overfitting loss=%.4f' % o_loss_, fontdict={'size': 20, 'color': 'red'})
        plt.text(0, -1.5, 'dropout loss=%.4f' % d_loss_, fontdict={'size': 20, 'color': 'blue'})
        plt.legend(loc='upper left')
        plt.ylim((-2.5, 2.5))
        plt.pause(0.1)

plt.ioff()
plt.show()