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
        """ Build the VGG16 net. """
        print('**********************\nBuild CNN\n**********************\n')

        config = self.config

        images = tf.placeholder(dtype = tf.float32, shape = [None, self.image_shape[0], self.image_shape[1], self.image_shape[2]])

        conv1_1_feats = self.nn.conv2d(images, 32, kernel_size = (7, 7), name = 'conv1_1')
        conv1_2_feats = self.nn.conv2d(conv1_1_feats, 32, kernel_size = (7, 7), name = 'conv1_2')        
        conv1_3_feats = self.nn.conv2d(conv1_2_feats, 32, kernel_size = (7, 7), name = 'conv1_3')
        pool1_feats = self.nn.max_pool2d(conv1_3_feats, name = 'pool1')

        print('shape of pool1_feats: ', pool1_feats.get_shape().as_list())

        conv2_1_feats = self.nn.conv2d(pool1_feats, 64,kernel_size = (5, 5), name = 'conv2_1')
        conv2_2_feats = self.nn.conv2d(conv2_1_feats, 64, kernel_size = (5, 5), name = 'conv2_2')
        conv2_3_feats = self.nn.conv2d(conv2_2_feats, 64, kernel_size = (5, 5), name = 'conv2_3')

        pool2_feats = self.nn.max_pool2d(conv2_3_feats, name = 'pool2')
        print('shape of pool2_feats: ', pool2_feats.get_shape().as_list())

        conv3_1_feats = self.nn.conv2d(pool2_feats, 128, kernel_size = (3, 3), name = 'conv3_1')
        conv3_2_feats = self.nn.conv2d(conv3_1_feats, 128, kernel_size = (3, 3), name = 'conv3_2')        
        conv3_3_feats = self.nn.conv2d(conv3_2_feats, 128, kernel_size = (3, 3), name = 'conv3_3')
        pool3_feats = self.nn.max_pool2d(conv3_3_feats, name = 'pool3')
        print('shape of pool3_feats: ', pool3_feats.get_shape().as_list())

        conv4_1_feats = self.nn.conv2d(pool3_feats, 256, kernel_size = (3, 3), name = 'conv4_1')
        conv4_2_feats = self.nn.conv2d(conv4_1_feats, 256, kernel_size = (3, 3), name = 'conv4_2')        
        conv4_3_feats = self.nn.conv2d(conv4_2_feats, 256, kernel_size = (3, 3), name = 'conv4_3')
        pool4_feats = self.nn.max_pool2d(conv4_3_feats, name = 'pool4')
        print('shape of pool4_feats: ', pool4_feats.get_shape().as_list())

        conv5_1_feats = self.nn.conv2d(pool4_feats, 256, kernel_size = (3, 3), name = 'conv5_1')
        conv5_2_feats = self.nn.conv2d(conv5_1_feats, 256, kernel_size = (3, 3), name = 'conv5_2')        
        conv5_3_feats = self.nn.conv2d(conv5_2_feats, 256, kernel_size = (3, 3), name = 'conv5_3')
        pool5_feats = self.nn.max_pool2d(conv5_3_feats, name = 'pool5')
        print('shape of pool5_feats: ', pool5_feats.get_shape().as_list())
        shapetemp = pool5_feats.get_shape().as_list()
        reshape_pool5_feats = tf.reshape(pool5_feats, [-1, shapetemp[1] * shapetemp[2] * shapetemp[3]])

        print('shape of reshape_pool5_feats: ', reshape_pool5_feats.get_shape().as_list())
        fc1 = self.nn.dense(reshape_pool5_feats, units = 1024)
        print('shape of fc1: ', fc1.get_shape().as_list())

        self.conv_feats = fc1
        self.images = images

    
    def build_rnn(self):
        """ Build the RNN. """
        print('**********************\nBuild RNN\n**********************\n')

        config = self.config

        # Setup the placeholders
        if self.is_train:
            # (32, 196, 512)
            contexts = self.conv_feats
            print('shape of contexts: ', contexts.get_shape().as_list())
            real_label = tf.placeholder(dtype = tf.int32, shape = [1, 2])
            self.real_label = real_label
       
        # Setup the LSTM
        lstm = tf.nn.rnn_cell.LSTMCell(
            config.num_lstm_units, #1024
            initializer = self.nn.fc_kernel_initializer)
        if self.is_train:
            lstm = tf.nn.rnn_cell.DropoutWrapper(
                lstm,
                input_keep_prob = 1.0-config.lstm_drop_rate,
                output_keep_prob = 1.0-config.lstm_drop_rate,
                state_keep_prob = 1.0-config.lstm_drop_rate)
        # print("output_size:",lstm.input_size)
        # print("state_size:",lstm.state_size)
        # ('output_size:', 512)
        # ('state_size:', LSTMStateTuple(c=512, h=512))


        # Initialize the LSTM using the mean context
        with tf.variable_scope("initialize"):
            #print "shape of conv_feats: ", self.conv_feats.get_shape()
            # (32, 196, 512)
            # context_mean = tf.reduce_mean(self.conv_feats, axis = 1)
            # print("shape of context_mean: ", context_mean.get_shape().as_list())
            # # (32, 512)
            initial_memory, initial_output = lstm.zero_state(1, tf.float32)
            print('shape of initial_memory: ', initial_memory.get_shape().as_list())
            print('shape of initial_output: ', initial_output.get_shape().as_list())
            initial_state = initial_memory, initial_output

        # Prepare to run
        predictions = []
        if self.is_train:
            alphas = []
            cross_entropies = []
            predictions_correct = []
            num_steps = contexts.get_shape().as_list()[0]
            if num_steps < 1:
                num_steps = 1
            last_output = initial_output
            last_memory = initial_memory
            # last_word = tf.zeros([config.batch_size], tf.int32)
        else:
            num_steps = 1
        last_state = last_memory, last_output
        
        outputs = []

        # Generate the words one by one
        for idx in range(num_steps):
           # Apply the LSTM
            with tf.variable_scope("lstm"):
                current_input = contexts[idx]
                current_input = tf.expand_dims(current_input, 0)
                # 32 1024
                print 'shape of current_input', current_input.get_shape().as_list()
                # print 'shape of last_state', last_state.get_shape()

                output, state = lstm(current_input, last_state)
                # print 'shape of output', output.get_shape()
                # (32, 512)
                # print 'shape of state', state
                memory, _ = state
                last_state = state

                outputs.append(output)

        print('shape of output: ', outputs[-1].get_shape().as_list())
        shapeofout = outputs[0].get_shape().as_list()
        outputs = tf.reshape(outputs, [1, shapeofout[0] * shapeofout[1]])
        logits = self.getLogits(outputs)
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
            last_output = output
            last_memory = memory
            last_state = state

        tf.get_variable_scope().reuse_variables()

        # Compute the final loss, if necessary
        if self.is_train:
            cross_entropy_loss = tf.reduce_mean(cross_entropies)
            self.cross_entropy_loss = cross_entropy_loss
            self.prediction = prediction
        else:
            self.initial_memory = initial_memory
            self.initial_output = initial_output
            self.last_memory = last_memory
            self.last_output = last_output
            self.memory = memory
            self.output = output
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
