import tensorflow as tf
import numpy as np

from base_model import BaseModel

class RCNNMODEL(BaseModel):
    def build(self):
        """ Build the model. """
        print('**********************\nBuild the model\n**********************\n')
        self.build_cnn()
        self.build_rnn()
        print(self.is_train)
        if self.is_train:
            self.build_optimizer()
            self.build_summary()

    def build_cnn(self):
        """ Build the CNN. """
        print("Building the CNN...")
        if self.config.cnn == 'mycnn':
            self.build_mycnn()
        else:
            self.build_resnet50()
        print("CNN built.")
    
    def build_mycnn(self):
        print('**********************\nBuild CNN\n**********************\n')

        config = self.config

        images = tf.placeholder(dtype = tf.float32, shape = [30, self.image_shape[0], self.image_shape[1], self.image_shape[2]])

        conv1_1_feats = self.nn.conv2d(images, 32, kernel_size = (7, 7), name = 'conv1_1')
        conv1_2_feats = self.nn.conv2d(conv1_1_feats, 32, kernel_size = (7, 7), name = 'conv1_2')        
        conv1_3_feats = self.nn.conv2d(conv1_2_feats, 32, kernel_size = (7, 7), name = 'conv1_3')
        conv1_3_feats = tf.nn.relu(conv1_3_feats)
        pool1_feats = self.nn.max_pool2d(conv1_3_feats, name = 'pool1')

        print('shape of pool1_feats: ', pool1_feats.get_shape().as_list())

        conv2_1_feats = self.nn.conv2d(pool1_feats, 64,kernel_size = (5, 5), name = 'conv2_1')
        conv2_2_feats = self.nn.conv2d(conv2_1_feats, 64, kernel_size = (5, 5), name = 'conv2_2')
        conv2_3_feats = self.nn.conv2d(conv2_2_feats, 64, kernel_size = (5, 5), name = 'conv2_3')
        conv2_3_feats = tf.nn.relu(conv2_3_feats)

        pool2_feats = self.nn.max_pool2d(conv2_3_feats, name = 'pool2')
        print('shape of pool2_feats: ', pool2_feats.get_shape().as_list())

        conv3_1_feats = self.nn.conv2d(pool2_feats, 128, kernel_size = (3, 3), name = 'conv3_1')
        conv3_2_feats = self.nn.conv2d(conv3_1_feats, 128, kernel_size = (3, 3), name = 'conv3_2')        
        conv3_3_feats = self.nn.conv2d(conv3_2_feats, 128, kernel_size = (3, 3), name = 'conv3_3')
        conv3_3_feats = tf.nn.relu(conv3_3_feats)

        pool3_feats = self.nn.max_pool2d(conv3_3_feats, name = 'pool3')
        print('shape of pool3_feats: ', pool3_feats.get_shape().as_list())

        conv4_1_feats = self.nn.conv2d(pool3_feats, 256, kernel_size = (3, 3), name = 'conv4_1')
        conv4_2_feats = self.nn.conv2d(conv4_1_feats, 256, kernel_size = (3, 3), name = 'conv4_2')        
        conv4_3_feats = self.nn.conv2d(conv4_2_feats, 256, kernel_size = (3, 3), name = 'conv4_3')
        conv4_3_feats = tf.nn.relu(conv4_3_feats)
        pool4_feats = self.nn.max_pool2d(conv4_3_feats, name = 'pool4')
        print('shape of pool4_feats: ', pool4_feats.get_shape().as_list())

        conv5_1_feats = self.nn.conv2d(pool4_feats, 256, kernel_size = (3, 3), name = 'conv5_1')
        conv5_2_feats = self.nn.conv2d(conv5_1_feats, 256, kernel_size = (3, 3), name = 'conv5_2')        
        conv5_3_feats = self.nn.conv2d(conv5_2_feats, 256, kernel_size = (3, 3), name = 'conv5_3')
        conv5_3_feats = tf.nn.relu(conv5_3_feats)

        pool5_feats = self.nn.max_pool2d(conv5_3_feats, name = 'pool5')
        print('shape of pool5_feats: ', pool5_feats.get_shape().as_list())
        shapetemp = pool5_feats.get_shape().as_list()
        reshape_pool5_feats = tf.reshape(pool5_feats, [-1, shapetemp[1] * shapetemp[2] * shapetemp[3]])

        print('shape of reshape_pool5_feats: ', reshape_pool5_feats.get_shape().as_list())
        fc1 = self.nn.dense(reshape_pool5_feats, units = 1024)
        fc1 = tf.nn.relu(fc1)
        fc1 = self.nn.dropout(fc1)
        print('shape of fc1: ', fc1.get_shape().as_list())

        fc2 = self.nn.dense(fc1, 512)
        fc2 = tf.nn.relu(fc2)
        fc2 = self.nn.dropout(fc2)
        print('shape of fc2: ', fc2.get_shape().as_list())

        self.conv_feats = fc2
        self.images = images

    def build_resnet50(self):
        """ Build the ResNet50. """
        print 'Buinding the ResNet50'
        config = self.config
        images = tf.placeholder(dtype = tf.float32, shape = [30, self.image_shape[0], self.image_shape[1], self.image_shape[2]])
        print 'shape of images: ', images.get_shape().as_list()

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
        print('shape of pool1_feats: ', pool1_feats.get_shape().as_list())

        res2a_feats = self.resnet_block(pool1_feats, 'res2a', 'bn2a', 64, 1)
        res2b_feats = self.resnet_block2(res2a_feats, 'res2b', 'bn2b', 64)
        res2c_feats = self.resnet_block2(res2b_feats, 'res2c', 'bn2c', 64)
        print('shape of res2c_feats: ', res2c_feats.get_shape().as_list())

        res3a_feats = self.resnet_block(res2c_feats, 'res3a', 'bn3a', 128)
        res3b_feats = self.resnet_block2(res3a_feats, 'res3b', 'bn3b', 128)
        res3c_feats = self.resnet_block2(res3b_feats, 'res3c', 'bn3c', 128)
        res3d_feats = self.resnet_block2(res3c_feats, 'res3d', 'bn3d', 128)
        print('shape of res3d_feats: ', res3d_feats.get_shape().as_list())

        res4a_feats = self.resnet_block(res3d_feats, 'res4a', 'bn4a', 256)
        res4b_feats = self.resnet_block2(res4a_feats, 'res4b', 'bn4b', 256)
        res4c_feats = self.resnet_block2(res4b_feats, 'res4c', 'bn4c', 256)
        res4d_feats = self.resnet_block2(res4c_feats, 'res4d', 'bn4d', 256)
        res4e_feats = self.resnet_block2(res4d_feats, 'res4e', 'bn4e', 256)
        res4f_feats = self.resnet_block2(res4e_feats, 'res4f', 'bn4f', 256)
        print('shape of res4f_feats: ', res4f_feats.get_shape().as_list())

        res5a_feats = self.resnet_block(res4f_feats, 'res5a', 'bn5a', 512)
        res5b_feats = self.resnet_block2(res5a_feats, 'res5b', 'bn5b', 512)
        res5c_feats = self.resnet_block2(res5b_feats, 'res5c', 'bn5c', 512)
        print('shape of res5c_feats: ', res5c_feats.get_shape().as_list())
        shapetemp = res5c_feats.get_shape().as_list()

        reshape_res5c_feats = tf.reshape(res5c_feats, [-1, shapetemp[1] * shapetemp[2] * shapetemp[3]])
        print('shape of reshape_res5c_feats: ', reshape_res5c_feats.get_shape().as_list())
        fc1 = self.nn.dense(reshape_res5c_feats, 2048)
        fc1 = self.nn.dropout(fc1)

        fc2 = self.nn.dense(fc1, 512)
        fc2 = self.nn.dropout(fc2)
        print('shape of fc2: ', fc2.get_shape().as_list())

        # global average pooling as final layer
        # shapetemp = res5c_feats.get_shape().as_list()
        # res5c_gap = tf.nn.avg_pool(res5c_feats, ksize = [1, shapetemp[1], shapetemp[2], 1], strides = [1, shapetemp[1], shapetemp[2], 1], padding = 'VALID')
        # print('shape of res5c_gap: ', res5c_gap.get_shape().as_list())
        # shapetemp = res5c_gap.get_shape().as_list()
        # reshape_res5c_gap = tf.reshape(res5c_gap, [-1, shapetemp[1] * shapetemp[2] * shapetemp[3]])
        # print('shape of reshape_res5c_gap: ', reshape_res5c_gap.get_shape().as_list())
        # fc1 = self.nn.dense(reshape_res5c_gap, 512)
        # fc1 = self.nn.dropout(fc1)
        # print('shape of fc1: ', fc1.get_shape().as_list())

        self.conv_feats = fc2
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

    def build_rnn(self):
        """ Build the RNN. """
        print('**********************\nBuild RNN\n**********************\n')

        config = self.config

        # Setup the placeholders
        if self.is_train:
            # (32, 196, 512)
            contexts = self.conv_feats
            print('shape of contexts: ', contexts.get_shape().as_list())
           # with tf.device("/cpu:0"):
            real_label = tf.placeholder(dtype = tf.int32, shape = [1, 2])
            self.real_label = real_label
       
        # Setup the LSTM
        layers = []
        for _ in range(config.num_layers):
            lstmlayer = tf.nn.rnn_cell.LSTMCell(
                config.num_lstm_units, #1024
                initializer = self.nn.fc_kernel_initializer)
            lstmlayer = tf.nn.rnn_cell.DropoutWrapper(
                lstmlayer,
                input_keep_prob = 1.0-config.lstm_drop_rate,
                output_keep_prob = 1.0-config.lstm_drop_rate,
                state_keep_prob = 1.0-config.lstm_drop_rate)
            layers.append(lstmlayer)

        lstm = tf.nn.rnn_cell.MultiRNNCell(layers, state_is_tuple = True) 
        # Initialize the LSTM using the mean context

        # Initialize the LSTM using the mean context
        with tf.variable_scope("initialize"):
            initial_state = lstm.zero_state(1, tf.float32)
        #     print('shape of initial_memory: ', initial_memory.get_shape().as_list())
        #     print('shape of initial_output: ', initial_output.get_shape().as_list())
        #     initial_state = initial_memory, initial_output

        # Prepare to run
        predictions = []
        if self.is_train:
            cross_entropies = []
            predictions_correct = []
            num_steps = contexts.get_shape().as_list()[0]
            print('steps: ', num_steps)
            if num_steps < 1:
                num_steps = 1
            last_state = initial_state
            # last_word = tf.zeros([config.batch_size], tf.int32)
        else:
            num_steps = 1
        last_state = initial_state
        
        outputs = []

        # Generate the words one by one
        for idx in range(num_steps):
           # Apply the LSTM
            with tf.variable_scope("lstm"):
                current_input = contexts[idx]
                current_input = tf.expand_dims(current_input, 0)
                # print 'shape of current_input', current_input.get_shape().as_list()
                output, state = lstm(current_input, last_state)
                last_state = state
                # print 'shape of output', output.get_shape().as_list()

                outputs.append(output)

        print('shape of output: ', outputs[-1].get_shape().as_list())
        shapeofout = outputs[0].get_shape().as_list()
        # outputs = tf.reshape(output[-1], [1,shapeofout[1]])
        logits = self.getLogits(outputs[-1])
        print('shape of logits: ',logits.get_shape().as_list())
        print('shape of real label: ',real_label.get_shape().as_list())

        if self.is_train:
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
                labels = real_label,
                logits = logits)
            cross_entropies.append(cross_entropy)
            probs = tf.nn.softmax(logits)
            prediction = tf.argmax(probs, 1)
            self.correct_pred = tf.equal(prediction, tf.argmax(real_label, 1))
            print('correct_pred: ',type(self.correct_pred))
            last_state = state

        tf.get_variable_scope().reuse_variables()

        # Compute the final loss, if necessary
        if self.is_train:
            cross_entropy_loss = tf.reduce_mean(cross_entropies)
            self.cross_entropy_loss = cross_entropy_loss
            self.prediction = prediction
        else:
            self.initial_state = initial_state
            self.last_state = last_state
            self.output = outputs[-1]
            self.probs = probs

        print("RNN built.")

    def getLogits(self, output):
        config = self.config
        if config.num_decode_layers == 1:
            # use 1 fc layer to decode
            logits = self.nn.dense(output,
                                   units = 2,
                                   activation = None,
                                   name = 'fc')
        else:
            print 'use 2 fc layers to decode'
            temp = self.nn.dense(output,
                                 units = config.dim_decode_layer,
                                 activation = tf.tanh,
                                 name = 'fc_1')
            temp = self.nn.dropout(temp)
            logits = self.nn.dense(temp,
                                   units = 2,
                                   activation = None,
                                   name = 'fc_2')
        return logits

    def build_optimizer(self):
        """ Setup the optimizer and training operation. """
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
                loss = self.cross_entropy_loss,
                global_step = self.global_step,
                learning_rate = learning_rate,
                optimizer = optimizer,
                clip_gradients = config.clip_gradients,
                learning_rate_decay_fn = learning_rate_decay_fn)

        self.opt_op = opt_op

    def build_summary(self):
        """ Build the summary (for TensorBoard visualization). """
        with tf.name_scope("variables"):
            for var in tf.trainable_variables():
                with tf.name_scope(var.name[:var.name.find(":")]):
                    self.variable_summary(var)

        with tf.name_scope("metrics"):
            tf.summary.scalar("cross_entropy_loss", self.cross_entropy_loss)

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
