import os
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
import copy
import json
from tqdm import tqdm

from utils.nn import NN
from utils.coco.coco import COCO
from utils.coco.pycocoevalcap.eval import COCOEvalCap
from utils.misc import ImageLoader, CaptionData, TopN

test_data = ['0013EDC2-8D7A-4A41-AEB5-D3BB592306D2',
'0030CBD1-2472-42C4-8CE4-E01A4E8E2F09',
'0036DF08-EEEC-467C-8CF1-5A54E0B13CE8',
'003D2553-266F-47E3-A420-F5B8F95217A7',
'0072E2C1-C395-409B-8078-365DD5C0513E',
'00C1E256-E3F7-431D-BCB6-DD1EF09E7DFE',
'00C79BB9-C2BC-4BF2-A8E6-5C3866DCBF3D',
'00CAD9C8-D90B-43C4-9964-79CAE6493DA7',
'00D5ADEF-A62D-4DF3-A9C0-31000C826023',
'00D6C763-CCE3-4D48-A165-324A36285DAB',
'00E8B6E7-9E4C-4FA9-B9C2-A9466667ACAC',
'00F9FF74-1D6D-4BF9-BB02-5C33DDADBE13',
'00FD9E8C-F811-477E-B42A-B6B44100CD5C',
'0113A22C-CDC6-48F9-ADC2-F2D38EA5FC90',
'0116B20D-36FA-4364-93C2-79DCCA2AB177',
'01347C43-CBFB-4EBB-99A3-CEA6332E7535',
'0152C839-EB23-41A3-9AA6-8847E54C937C',
'01665898-4C2C-4201-B889-D303ACC8D045',
'016FC8F3-D808-4AFC-85F2-3DF314B65B78',
'01922428-C7A9-45D1-922A-164C1394C970',
'019C6DF4-30C1-4EBB-9904-713BBDA77F0D',
'01A0F362-15F9-4344-9AB9-B961A0FC4336',
'01C2E827-0665-4C99-976D-757E5B9D0F3C',
'01D03196-A1C9-4EA8-A729-8896EF4DB348',
'01F14201-7DC3-4F27-9F93-950E17B700B9',
'02038AEF-061B-43B1-A6CE-F236F9BB54EA',
'02400A3A-B9CC-42DA-92FF-98D3780A61F6',
'0245345F-733A-4BE8-887B-BBE122DBECCC',
'02459974-2670-4928-9FBD-99C658E85F19',
'02647665-2213-4E4F-AA33-2E4B91BA270A']

class BaseModel(object):
    def __init__(self, config):
        self.config = config
        self.is_train = True if config.phase == 'train' else False
        self.train_cnn = self.is_train and config.train_cnn
        self.image_loader = ImageLoader('./utils/ilsvrc_2012_mean.npy')
        self.image_shape = [512, 512, 1]
        self.nn = NN(config)
        self.global_step = tf.Variable(0,
                                       name = 'global_step',
                                       trainable = False)
        self.build()

    def build(self):
        raise NotImplementedError()

    def train(self, sess, train_data, train_data_label, dataobj, transign):
        """ Train the model. """
        print("Training the model...")
        config = self.config

        if not os.path.exists(config.summary_dir):
            os.mkdir(config.summary_dir)
        train_writer = tf.summary.FileWriter(config.summary_dir,
                                             sess.graph)

        record = open('lossrecord.txt', 'w')
        for _ in tqdm(list(range(config.num_epochs)), desc='epoch'):

            for onepatient in tqdm(train_data[31:], desc = 'data'):
                onelabel = self.getLabel(onepatient, train_data_label)
                slices = dataobj.getOnePatient(onepatient, transign)
                # print('shape: ', slices.shape)
                feed_dict = {self.images: slices, self.real_label: onelabel}            

                _, summary, global_step = sess.run([self.opt_op,
                                                    self.summary,
                                                    self.global_step,],
                                                    feed_dict=feed_dict)

                if (global_step + 1) % config.save_period == 0:
                    self.save()
                if (global_step + 1) % 1000 == 0:
                    print('@@@@@@@@@@@@@@@@@@@@@@@@@')
                    temploss = 0.0
                    correctpercent = 0.0
                    for testpatient in train_data[:31]:
                        testlabel = self.getLabel(testpatient, train_data_label)
                        testslices = dataobj.getOnePatient(testpatient, False)
                        feed_dict = {self.images: testslices, self.real_label: testlabel}
                        loss, correctpred = sess.run([self.cross_entropy_loss, self.correct_pred], feed_dict=feed_dict)
                        temploss += loss
                        if correctpred == True:
                            correctpercent += 1
                    print('loss: ', temploss / 30)
                    print('correction: ', correctpercent / 30)
                
                    record.write('loss: ' + str(temploss / 30) + '\n')
            train_writer.add_summary(summary, global_step)

        self.save()
        record.close()
        train_writer.close()
        print("Training complete.")

    def getLabel2(self, onepatient, alllabel):
        for onelabel in alllabel:
            if onepatient == onelabel[0]:
                if int(onelabel[1]) == 1:
                    return([0, 1])
                elif int(onelabel[1]) == 0:
                    return([1, 0])

                break
            
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

    def test(self, sess, testdatapath, dataobj):
        """ Test the model using any given images. """
        print("Testing the model ...")
        config = self.config

        testpatients = os.listdir(testdatapath)
        predrecord = open('predrecord.csv', 'w')
        
        predrecord.write('id,ret\n')

        for onetestpatient in tqdm(testpatients):
            onelabel = [[0, 0]]
            slices = dataobj.getOnePatient2(testdatapath, onetestpatient)
            # print('shape: ', slices.shape)
            
            feed_dict = {self.images: slices, self.real_label: onelabel}
            prediction = sess.run([self.prediction], feed_dict=feed_dict)
            # print(type(prediction))
            # print(prediction[0])
            predrecord.write(onetestpatient + ',' + str(prediction[0][0]) + '\n')
            
        predrecord.close()
        print("Test completed.")

    def save(self):
        """ Save the model. """
        config = self.config
        data = {v.name: v.eval() for v in tf.global_variables()}
        save_path = os.path.join(config.save_dir, str(self.global_step.eval()))

        print((" Saving the model to %s..." % (save_path+".npy")))
        np.save(save_path, data)
        info_file = open(os.path.join(config.save_dir, "config.pickle"), "wb")
        config_ = copy.copy(config)
        config_.global_step = self.global_step.eval()
        pickle.dump(config_, info_file)
        info_file.close()
        print("Model saved.")

    def load(self, sess, model_file=None):
        """ Load the model. """
        config = self.config
        if model_file is not None:
            save_path = model_file
        else:
            info_path = os.path.join(config.save_dir, "config.pickle")
            info_file = open(info_path, "rb")
            config = pickle.load(info_file)
            global_step = config.global_step
            info_file.close()
            save_path = os.path.join(config.save_dir,
                                     str(global_step)+".npy")

        print("Loading the model from %s..." %save_path)
        data_dict = np.load(save_path).item()
        count = 0
        for v in tqdm(tf.global_variables()):
            if v.name in data_dict.keys():
                sess.run(v.assign(data_dict[v.name]))
                count += 1
        print("%d tensors loaded." %count)

    def load_cnn(self, session, data_path, ignore_missing=True):
        """ Load a pretrained CNN model. """
        print("Loading the CNN from %s..." %data_path)
        data_dict = np.load(data_path).item()
        count = 0
        for op_name in tqdm(data_dict):
            with tf.variable_scope(op_name, reuse = True):
                for param_name, data in data_dict[op_name].iteritems():
                    try:
                        var = tf.get_variable(param_name)
                        session.run(var.assign(data))
                        count += 1
                    except ValueError:
                        pass
        print("%d tensors loaded." %count)
