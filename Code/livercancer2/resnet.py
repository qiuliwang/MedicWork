import tensorflow as tf
import numpy as np

from base_model import BaseModel
from tqdm import tqdm

class RCNNMODEL(BaseModel):
    def build(self):
        print('*******************\nBuilding the Model\n*******************')
        self.build_resnet50()
        print('cnn built')
        self.buildmodel()
        self.build_optimizer()
        self.build_summary()
    
    def buildmodel(self):
        self.labels = tf.placeholder(dtype = tf.float32, shape = [None, 2])
        logits = self.getLogits(self.conv_feats)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
                labels = self.labels,
                logits = logits)
        
        self.loss_of_entropy = tf.reduce_mean(cross_entropy)
        probs = tf.nn.softmax(logits)
        prediction = tf.argmax(probs, 1)
        correct_prediction = tf.equal(tf.argmax(probs, 1), tf.argmax(self.labels, 1))
        self.accruacy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def getLogits(self, output):
        config = self.config
        if config.num_decode_layers == 1:
            # use 1 fc layer to decode
            logits = self.nn.dense(output,
                                units = 2,
                                activation = None)
        else:
            temp = self.nn.dense(output,
                                units = config.dim_decode_layer,
                                activation = tf.tanh)
            temp = self.nn.dropout(temp)
            logits = self.nn.dense(temp,
                                units = 2,
                                activation = None)

        return logits
        
    def getLabel(self, onepatient, alllabel):
        label = []
        for onelabel in alllabel:
            if onepatient == onelabel[0]:
                if int(onelabel[1]) == 1:
                    label.append([0, 1])
                elif int(onelabel[1]) == 0:
                    label.append([1, 0])

                break
            
        return label




    def build_resnet50(self):
        """ Build the ResNet50. """
        # print('Buinding the ResNet50')
        config = self.config
        images = tf.placeholder(dtype = tf.float32, shape = [None, self.image_shape[0], self.image_shape[1], self.image_shape[2]])
        conv1_feats = self.nn.conv2d(images,
                                  filters = 64,
                                  kernel_size = (7, 7),
                                  strides = (2, 2),
                                  activation = None,
                                  name = 'conv1')
        conv1_feats = self.nn.batch_norm(conv1_feats, 'bn_conv1')
        conv1_feats = tf.nn.relu(conv1_feats)
        pool1_feats = self.nn.max_pool2d(conv1_feats,
                                      pool_size = (3, 3),
                                      strides = (2, 2),
                                      name = 'pool1')
        # print('shape of pool1_feats: ', pool1_feats.get_shape().as_list())

        res2a_feats = self.resnet_block(pool1_feats, 'res2a', 'bn2a', 64, 1)
        res2b_feats = self.resnet_block2(res2a_feats, 'res2b', 'bn2b', 64)
        res2c_feats = self.resnet_block2(res2b_feats, 'res2c', 'bn2c', 64)
        # print('shape of res2c_feats: ', res2c_feats.get_shape().as_list())

        res3a_feats = self.resnet_block(res2c_feats, 'res3a', 'bn3a', 128)
        res3b_feats = self.resnet_block2(res3a_feats, 'res3b', 'bn3b', 128)
        res3c_feats = self.resnet_block2(res3b_feats, 'res3c', 'bn3c', 128)
        res3d_feats = self.resnet_block2(res3c_feats, 'res3d', 'bn3d', 128)
        # print('shape of res3d_feats: ', res3d_feats.get_shape().as_list())

        res4a_feats = self.resnet_block(res3d_feats, 'res4a', 'bn4a', 256)
        res4b_feats = self.resnet_block2(res4a_feats, 'res4b', 'bn4b', 256)
        res4c_feats = self.resnet_block2(res4b_feats, 'res4c', 'bn4c', 256)
        res4d_feats = self.resnet_block2(res4c_feats, 'res4d', 'bn4d', 256)
        res4e_feats = self.resnet_block2(res4d_feats, 'res4e', 'bn4e', 256)
        res4f_feats = self.resnet_block2(res4e_feats, 'res4f', 'bn4f', 256)
        # print('shape of res4f_feats: ', res4f_feats.get_shape().as_list())

        res5a_feats = self.resnet_block(res4f_feats, 'res5a', 'bn5a', 512)
        res5b_feats = self.resnet_block2(res5a_feats, 'res5b', 'bn5b', 512)
        res5c_feats = self.resnet_block2(res5b_feats, 'res5c', 'bn5c', 512)
        shapetemp = res5c_feats.get_shape().as_list()

        res5c_gap = tf.nn.avg_pool(res5c_feats, ksize = [1, shapetemp[1], shapetemp[2], 1], strides = [1, shapetemp[1], shapetemp[2], 1], padding = 'VALID')

        # print('shape of res5c_gap: ', res5c_gap.get_shape().as_list())
        shapetemp = res5c_gap.get_shape().as_list()

        reshape_res5c_gap = tf.reshape(res5c_gap, [-1, shapetemp[1] * shapetemp[2] * shapetemp[3]])
        # print('shape of reshape_res5c_gap: ', reshape_res5c_gap.get_shape().as_list())

        fc1 = self.nn.dense(reshape_res5c_gap, 512)
        fc1 = self.nn.dropout(fc1, name = 'fc1')
        # print('shape of fc1: ', fc1.get_shape().as_list())

        self.conv_feats = fc1
        self.images = images

    def resnet_block(self, inputs, name1, name2, c, s=2):
        """ A basic block of ResNet. """
        branch1_feats = self.nn.conv2d(inputs,
                                    filters = 4*c,
                                    kernel_size = (1, 1),
                                    strides = (s, s),
                                    activation = None,
                                    use_bias = False,
                                    name = name1+'_branch1')
        branch1_feats = self.nn.batch_norm(branch1_feats, name2+'_branch1')

        branch2a_feats = self.nn.conv2d(inputs,
                                     filters = c,
                                     kernel_size = (1, 1),
                                     strides = (s, s),
                                     activation = None,
                                     use_bias = False,
                                     name = name1+'_branch2a')
        branch2a_feats = self.nn.batch_norm(branch2a_feats, name2+'_branch2a')
        branch2a_feats = tf.nn.relu(branch2a_feats)

        branch2b_feats = self.nn.conv2d(branch2a_feats,
                                     filters = c,
                                     kernel_size = (3, 3),
                                     strides = (1, 1),
                                     activation = None,
                                     use_bias = False,
                                     name = name1+'_branch2b')
        branch2b_feats = self.nn.batch_norm(branch2b_feats, name2+'_branch2b')
        branch2b_feats = tf.nn.relu(branch2b_feats)

        branch2c_feats = self.nn.conv2d(branch2b_feats,
                                     filters = 4*c,
                                     kernel_size = (1, 1),
                                     strides = (1, 1),
                                     activation = None,
                                     use_bias = False,
                                     name = name1+'_branch2c')
        branch2c_feats = self.nn.batch_norm(branch2c_feats, name2+'_branch2c')

        outputs = branch1_feats + branch2c_feats
        outputs = tf.nn.relu(outputs)
        return outputs

    def resnet_block2(self, inputs, name1, name2, c):
        """ Another basic block of ResNet. """
        branch2a_feats = self.nn.conv2d(inputs,
                                     filters = c,
                                     kernel_size = (1, 1),
                                     strides = (1, 1),
                                     activation = None,
                                     use_bias = False,
                                     name = name1+'_branch2a')
        branch2a_feats = self.nn.batch_norm(branch2a_feats, name2+'_branch2a')
        branch2a_feats = tf.nn.relu(branch2a_feats)

        branch2b_feats = self.nn.conv2d(branch2a_feats,
                                     filters = c,
                                     kernel_size = (3, 3),
                                     strides = (1, 1),
                                     activation = None,
                                     use_bias = False,
                                     name = name1+'_branch2b')
        branch2b_feats = self.nn.batch_norm(branch2b_feats, name2+'_branch2b')
        branch2b_feats = tf.nn.relu(branch2b_feats)

        branch2c_feats = self.nn.conv2d(branch2b_feats,
                                     filters = 4*c,
                                     kernel_size = (1, 1),
                                     strides = (1, 1),
                                     activation = None,
                                     use_bias = False,
                                     name = name1+'_branch2c')
        branch2c_feats = self.nn.batch_norm(branch2c_feats, name2+'_branch2c')

        outputs = inputs + branch2c_feats
        outputs = tf.nn.relu(outputs)
        return outputs

    def build_optimizer(self):
        """ Setup the optimizer and training operation. """
        print('Setup the optimizer and training operation.')
        config = self.config

        learning_rate = tf.constant(config.initial_learning_rate)
        if config.learning_rate_decay_factor < 1.0:
            def _learning_rate_decay_fn(learning_rate, global_step):
                return tf.train.exponential_decay(
                    learning_rate,
                    global_step,
                    decay_steps = config.num_steps_per_decay,
                    decay_rate = config.learning_rate_decay_factor,
                    staircase = True)
            learning_rate_decay_fn = _learning_rate_decay_fn
        else:
            learning_rate_decay_fn = None

        with tf.variable_scope('optimizer', reuse = tf.AUTO_REUSE):
            if config.optimizer == 'Adam':
                optimizer = tf.train.AdamOptimizer(
                    learning_rate = config.initial_learning_rate,
                    beta1 = config.beta1,
                    beta2 = config.beta2,
                    epsilon = config.epsilon
                    )
            elif config.optimizer == 'RMSProp':
                optimizer = tf.train.RMSPropOptimizer(
                    learning_rate = config.initial_learning_rate,
                    decay = config.decay,
                    momentum = config.momentum,
                    centered = config.centered,
                    epsilon = config.epsilon
                )
            elif config.optimizer == 'Momentum':
                optimizer = tf.train.MomentumOptimizer(
                    learning_rate = config.initial_learning_rate,
                    momentum = config.momentum,
                    use_nesterov = config.use_nesterov
                )
            else:
                optimizer = tf.train.GradientDescentOptimizer(
                    learning_rate = config.initial_learning_rate
                )

            opt_op = tf.contrib.layers.optimize_loss(
                loss = self.loss_of_entropy,
                global_step = self.global_step,
                learning_rate = learning_rate,
                optimizer = optimizer,
                clip_gradients = config.clip_gradients,
                learning_rate_decay_fn = learning_rate_decay_fn)

        self.opt_op = opt_op

    def build_summary(self):
        """ Build the summary (for TensorBoard visualization). """
        print('Build the summary (for TensorBoard visualization)')
        with tf.name_scope("variables"):
            for var in tf.trainable_variables():
                with tf.name_scope(var.name[:var.name.find(":")]):
                    self.variable_summary(var)

        with tf.name_scope("metrics"):
            tf.summary.scalar("loss_of_entropy", self.loss_of_entropy)

        self.summary = tf.summary.merge_all()

    def variable_summary(self, var):
        """ Build the summary for a variable. """
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)
