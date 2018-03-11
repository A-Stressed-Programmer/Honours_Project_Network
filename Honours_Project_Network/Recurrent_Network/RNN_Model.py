#Project dependancies
#--Project Imports--
#Tensorflow
import tensorflow as tf
from tensorflow.contrib import rnn
#Numpy
import numpy as np
from numpy import genfromtxt
from numpy import array
#OS
import sys
import math
#Import Plotter
import matplotlib.pyplot as plt

#Project Dependancies Imports
from File_Reader import File_Reader
filereader = File_Reader()#Declare local
from Data_Parser import Data_Parser
dataparser = Data_Parser()#Declare local
from Data_Plotter import Data_Plotter
dataplotter = Data_Plotter()#Declare local

#Main Class
class RNN():
    '''
    RNN Class housing Construction definitions for the main RNN inside Tensorflow

    Initialize main variables for RNN Construction;
    RNN(self, n_neurons, n_layers, learning_rate, fn_select, filename, epoch)
    '''
    def __init__(self, n_neurons=int, n_layers=int, learning_rate=float, fn_select="", filename="", epoch=int, batches=int, percentage=float, toggle_num=int):
        '''
        Initialize main variables for RNN Construction;
        RNN(n_neurons, n_layers, learning_rate, fn_select, filename, epoch, batches, percentage)

        n_neurons = Number of Neurons in A Layer
        n_layers = Number of Layers
        learning_rate = The learning rate of the error for gradient descent
        fn_select = Activation Function Selector
        filename = Name of File inside the "Datasets" Folder
        epoch = Total cycle count of network
        batches = Typically One for Now
        percentage = The split of Training and Testing Data
        toggle = Toggle selection for dataset switch
        '''
        #User Input feedback
        print("---Honours Neural Network Project---", "\n", "----INITIALIZE!----", "\n")
        print("VALUES: \n", "Number of Neurons:[", n_neurons,"] \n Number of Layers:[",n_layers ,"] \n Learning Rate:[", learning_rate, "] \n Function Selection:[", fn_select, "] \n Filename:[", filename,"] \n Epoch:[",epoch,"] \n Number of Batches:[",batches,"] \n")

        #--Main variables for RNN--
        #USER DEFINED VARIABLES;
        self.n_neurons = n_neurons#Neurons in each layer
        self.n_layers = n_layers#Number of Layers
        self.learning_rate = learning_rate#Learning rate for main network
        self.filename = filename#Store filename for reading
        self.epoch = epoch#Number of EPOCH runs
        self.num_batch = batches#Get number of batches  int(round(math.sqrt(self.num_data)))
        self.n_outputs =1#Neural Network Output

        #AUTOMATIC VARIABLES
        self.activation_function = RNN.get_activation_fn(fn_select)#Get Function
        self.x_axis_plotter_data, self.input_data, self.label_data = filereader.parse_data(self.filename)#GET DATA

        def auto_data_switch(toggle):
            '''
            Toggle Data switch will swap injection data from;
       
            [Open][High][Low]-->[Projection CONCAT]-->[Predictied Close] 
             to
            [Close]-->[Projection CONCAT]-->[Preicted Next Close]
            '''
            if toggle == 1:
                '''
                Toggle One, [Open][High][Low]-->[Projection CONCAT]-->[Predictied Close] 
                '''
                print("Toggle One Selected; \n [Open][High][Low]-->[Projection CONCAT]-->[Predictied Close] \n")
                #Ajust Tensor data
                self.n_inputs = 3
                #Get Data
                self.x_axis_plotter_data, open_high_low_data, closing_data = filereader.parse_data(self.filename)#GET DATA

                #Declare Local
                input_data = open_high_low_data
                self.input_data = open_high_low_data
                label_data = closing_data
                #Return Sets
                return input_data, label_data
            if toggle == 2:
                '''
                Toggle Two, [Close]-->[Projection CONCAT]-->[Preicted Next Close]
                '''
                print("Toggle Two Selected; \n [Close]-->[Projection CONCAT]-->[Preicted Next Close] \n")
                #Ajust Tensor data
                self.n_inputs = 1
                #Get Data
                self.x_axis_plotter_data, open_high_low_data, closing_data = filereader.parse_data(self.filename)#GET DATA

                #Declare Local
                input_data = closing_data[:-1]
                label_data = filereader.get_targets(closing_data)#Warning Pops First value of array structure, WILL CAUSE TENSOR VALUE ERROR
                #Return Sets
                return input_data, label_data
            else:
                '''
                Default
                Toggle Three, [Open][High][Low]-->[Projection CONCAT]-->[Predictied Close] 
                '''
                print("Toggle DEFAULT Selected; \n [Open][High][Low]-->[Projection CONCAT]-->[Predictied Close] \n")
                #Ajust Tensor data
                self.n_inputs = 3
                #Get Data
                self.x_axis_plotter_data, open_high_low_data, closing_data = filereader.parse_data(self.filename)#GET DATA

                #Declare Local
                input_data, self.input_data = open_high_low_data
                label_data = closing_data
                #Return Sets
                return input_data, label_data
                ValueError("Incorrect Toggle Select, 1-2")

        #Insert Auto Here(WARNING: MANUAL OVERRIDE OF MAIN TENSOR VALUES HERE)
        print("\n(WARNING: MANUAL OVERRIDE OF MAIN TENSOR VALUES HERE)", "\n Toggle Select:[", toggle_num, "]\n")
        t_inject_input, t_inject_label = auto_data_switch(toggle_num)

        temp_input, temp_label = dataparser.normalize_data(t_inject_input, t_inject_label, self.n_inputs)#NORMALIZE DATA
        self.train_data, self.train_label, self.test_data, self.test_label = RNN.split_data(temp_input, temp_label, percentage)#Split dataset into Train/Test

        #Save dataset to file
        print("\n Saving Training & Testing Data to files:")
        filereader.read_into_temp(self.train_data, '/Data/train_data.csv')
        filereader.read_into_temp(self.train_label, '/Data/train_label.csv')
        filereader.read_into_temp(self.test_data, '/Data/test_data.csv')
        filereader.read_into_temp(self.test_label, '/Data/test_label.csv')
        print("File Saved Successfully!\n")

        #Size Data
        self.train_size = len(self.train_data)#Train INPUT Data
        self.train_label_size = len(self.train_label)#Train LABEL Data
        self.test_size = len(self.test_data)#Test INPUT Data
        self.test_label_size = len(self.test_label)#Test LABEL Data
        #Sort Steps
        self.train_step = int(round(self.train_size/self.num_batch))
        self.test_step = int(round(self.test_size/self.num_batch))

        #Session Storage
        self.error_loss = []
        self.train_results = []
        self.test_results = []
        #For Bidirectional Networks
        self.fw_results = []
        self.bw_results = []

        self.current_pos = 0#Zero position for batching
        self.t_backup = 0#Backup for [Index out of range]

        #Plotter
        #dataplotter.input_graph(self.label_data, self.filename)

        #Return Feedback for eval
        print("\n", "#--Network Variables--#")
        print("#--DATASET DETAILS(", filename, "): \n [Inputs(",self.n_inputs,")] \n [Outputs(", self.n_outputs, ")] \n [Size(", len(self.input_data), ")]")
        print("#--TRAINING--#", "\n Input Data Size:[",self.train_size,"]\n Number of Steps:[", self.train_step,"]\n Number of Batches:[", self.num_batch,"]")
        print("#--TESTING--#", "\n Input Data Size:[",self.test_size,"]\n Number of Steps:[", self.test_step,"]\n")
    
    def split_data(input_data, label_data, percentage):
        '''
        Split Input dataset into Training(LEFT) and Testing(RIGHT) set for RNN model in format

        split_data(input_data, label_data, percentage)

        return train_data, train_label, test_data, test_label
        '''
        #Temp Store
        data = []
        label = []
        data = input_data
        label = label_data

        #Local Variables
        #--Training
        train_data= np.array
        train_label = np.array
        #--Testing
        test_data = np.array
        test_label = np.array

        #--Training--#
        train_data = data[int(len(data) * .0) : int(len(data) * percentage)]#Return sets from 0 to Percentage
        train_label = label[int(len(label) * .0) : int(len(label) * percentage)]#Return sets from 0 to Percentage
        #--Testing--#
        test_data = data[int(len(data) * percentage) : int(len(data) * 1)]#Returns sets from Percentage to 1
        test_label = label[int(len(label) * percentage) : int(len(label) * 1)] #Returns sets from Percentage to 1
        #Return Train_data, train_label, Test_data, test_label
        return train_data, train_label, test_data, test_label

    def get_activation_fn(type):#Translate type and return activation function for network
        '''
        Select Activation functions automatically, Sigmoid(sig), Tahn(tahn), Softsign(softsign), relu(relu), default_defactor = relu
        '''
        print("Activation Select: [",type,"]")#Prompter
        activation = ''#Empty var holder
        #Main Loop

        if type == 'sig':#SIGMOID
            #Return Sigmoid
            '''
            f(x)=1/1+e^-x
            '''
            print("Sigmoid: [f(x)=1/1+e^-x] Selected!")
            activation = tf.sigmoid

        elif type == 'tahn':#TAHN
            #Return Tahn
            '''
            f(x)=tanh(x)=e^x - e^-x/e^x + e^-x
            '''
            print("Tahn: [f(x)=tanh(x)=e^x - e^-x/e^x + e^-x] Selected!")
            activation = tf.tanh

        elif type == 'softsign':#SOFTSIGN
            #Return Softsign
            '''
            f(x)=x/1+[X]
            '''
            print("Softsign Selected!")
            activation = tf.nn.softsign

        elif type == 'relu':#RECTIFIED LINEAR UNIT
            #Return Relu
            '''
            Rectified Linear Unit(ReLU)
            f(x)={0 for x < 0
                 {x for x >/ 0
            '''
            print("Rectified Linear Unit(ReLU) Selected!")
            activation = tf.nn.relu

        else:#EXCEPTION
            #Return default
            print("DEFAULT SETTINGS!")#CHANGE ME
            activation = tf.nn.selu

        return activation#Return Activation Function
        print("Activation Retured to main task!")#Conformation of task completion
        #--WARNING ONLY RELU WORKS DUE TO UN-NORMALIZED DATA--#

    #--------------------------#
    #Recurrent Neural Networks# #UPDATE#

    def train_rnn_model(self):
        '''
        Train Deep RNN Cell model, Recurrance within the main cell body
        '''
        tf.reset_default_graph()#Reset

        #--TOP HEAVY VARIABLES--#
        dataset_x = self.train_data#Data
        dataset_y = self.train_label#Label
        steps = self.train_step#Step Number
        num_in = self.n_inputs#Number of Inputs
        num_out = self.n_outputs#Number of Outputs
        prob = 1.0#Keep Prop for Outputs of Neural Network
        num_data = self.train_size#Size of Network

        #Embedded Batcher
        def next_batch():
            '''
            Internal Next Training Batcher for input data into network
            '''
            #Local Variables
            x_batch = []
            y_batch = []
            issue_warn = False

            #The lord have mercy upon this horrific code:
            #Checker
            temp_checker = self.current_pos + steps
            if temp_checker > num_data:
                #Issue
                issue_warn = True
            else:
                #No Issue
                issue_warn = False

            #Main loop
            if self.current_pos == 0:#If Started
                t_min = 0#Set Min
                t_max = t_min + steps#Set Max
                idx = np.arange(t_min,t_max)
                x_batch = [dataset_x[i] for i in idx]
                y_batch = [dataset_y[i] for i in idx]
                #Position Index in statements
                self.current_pos = self.current_pos * 0#Reset
                self.current_pos = self.current_pos + t_max#Set for next

            elif issue_warn == False and self.current_pos != 0:#Regular Batch
                t_min = self.current_pos#Set Min
                t_max = t_min + steps#Set Max
                idx = np.arange(t_min,t_max)
                x_batch = [dataset_x[i] for i in idx]
                y_batch = [dataset_y[i] for i in idx]

                #Position Index in statements
                self.current_pos = self.current_pos * 0#Reset
                self.current_pos = self.current_pos + t_max#Set for next

                #OLD
                self.t_backup = self.t_backup * 0#Reset
                self.t_backup = self.current_pos#Setnew

            elif issue_warn == True and self.current_pos != 0:#If T_Max over limit
                #t_min = self.t_backup OLD
                t_min = num_data - steps
                t_max = num_data#Set Maximum
                idx = np.arange(t_min,t_max)
                x_batch = [dataset_x[i] for i in idx]
                y_batch = [dataset_y[i] for i in idx]

            #Reshape
            x_batch = array(x_batch).reshape(1,steps,num_in)
            y_batch = array(y_batch).reshape(steps,num_out)
            #Return Batching
            return x_batch, y_batch

        X = tf.placeholder(tf.float32, shape=[1, steps, num_in], name="x")#INPUT_BATCH_DATA_PLACEHOLDER Rank 3 Matrix
        Y = tf.placeholder(tf.float32, shape=[steps, num_out], name="y")#TARGET_BATCH_DATA_PLACEHOLDER

        '''#OLD
        #Layer declare: Layer[(Number of cells, What Activation FN)Number of layers]
        layers = [
            tf.contrib.rnn.BasicRNNCell(num_units=self.n_neurons, activation=self.activation_function)
            for layer in range(self.n_layers)]

        #Dropper layer for over saturated data
        cells = [tf.contrib.rnn.DropoutWrapper(cell,input_keep_prob=prob, output_keep_prob=prob, state_keep_prob=prob) for cell in layers]
        '''
        #Main RNN Framework#
        rnn_cell=[]
        for _ in range(self.n_layers):
            cell = tf.contrib.rnn.BasicRNNCell(num_units=self.n_neurons, activation=self.activation_function)#Layer declare: Layer[(Number of cells, What Activation FN)Number of layers]
            cell = tf.nn.rnn_cell.DropoutWrapper(cell,input_keep_prob=prob, output_keep_prob=prob, state_keep_prob=prob)#Dropper layer for over saturated data
            rnn_cell.append(cell)#Append to rnn_cell

        multi_layer_cell = tf.contrib.rnn.MultiRNNCell(rnn_cell)#Append cell to layers, Stack cells to layers
        rnn_outputs, states = tf.nn.dynamic_rnn(multi_layer_cell, X, dtype=tf.float32)#Declare outputs and states for loss

        #OUTPUT PROJECT WRAPPER#
        with tf.name_scope("wrapper"):
            '''
            Output Projection Wrapper
            '''
            stacked_rnn_outputs = tf.reshape(rnn_outputs, [-1, self.n_neurons])
            stacked_outputs = tf.layers.dense(stacked_rnn_outputs, num_out)
            outputs = tf.reshape(stacked_outputs, [steps, num_out])

        #LOSS#
        with tf.name_scope("loss"):
            loss = tf.reduce_sum(tf.square(outputs - Y))#Sum of Loss
            optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate)#OPTIMIZER, ADAM is MOST ACCURATE MODEL #Change me to Automatic selection
            '''
            Adam Optimizer is an extenstion to Stochastic Gradient Descent. Designed by Diderik Kingma from OpenAI and Jimmy Ba from Tornoto from;
            “Adam: A Method for Stochastic Optimization“:https://arxiv.org/abs/1412.6980;

            Learning_Rate /or/ Alpha = Is the Porportion that weights are updated(0.0001 for smaller variables, 0.3 for larger)
            beta1 = Exponential decay rate for the first moment estimates
            beta2 = Exopential decay rate for second moment estimates
            epsilon = Number to prevent division by zero, DO NOT MODIFY
            '''
            training_op = optimizer.minimize(loss)

        #Initialize Variables
        init = tf.global_variables_initializer()
        #Create Saver
        saver = tf.train.Saver()

        #--Main Session--#
        with tf.Session() as sess:
            init.run()#Initialize
            #---Outer Epoch Shell---#
            print("---Training RNN Model---")
            for epoch  in range(self.epoch):
                #---Inner Iteration shell----#
                self.train_results = []*0#Zero Results to grab last epoch(Most Accurate)
                for iteration in range(self.num_batch):
                    #---Execution Shell---#
                    x_batch, y_batch = next_batch()#Train           
                    _, output_val, mse = sess.run([training_op, outputs, loss], feed_dict={X: x_batch, Y: y_batch})#TRAINING
                    results_data = []#Append data to array
                    for i in range(len(output_val)):
                        results_data.append(output_val[i])
                    self.train_results.append(results_data)#Append Results
                    #---Execution Shell End---#
                #---Inner Shell End---#
                print("Training RNN: Epoch: [",epoch ,"/" , self.epoch,"] ", "MSE: [",mse,"]")#Prompter ,"Output: [", output_val, "]"
                error = str(mse)
                self.error_loss.append(error)
            #---Outer Shell End---#
            print("---Training Completed!---")
            print("Saving!")
            saver.save(sess, "Recurrent_Network/Saves/RNN/Recurrent_Neural_Network.ckpt") 

            #File Printers#
            print("Printing to Files")
            filereader.read_into_temp(np.ravel(self.train_results), "/RNN/train_results.csv")#Save to file
            filereader.read_into_temp(self.error_loss, "/RNN/error_results.csv")#Save to File
            print("Task Completed!")

            #Plotters#
            print("Plotting...")
            dataplotter.train_graph(self.filename,np.ravel(self.train_results), 'rnn')
            dataplotter.plt_error_graph(self.filename, np.ravel(self.error_loss), self.epoch, 'rnn')
            dataplotter.c_train_graph(self.filename,np.ravel(self.train_results), np.ravel(self.train_label), 'rnn')
            print("Plot Completed!")
            #---END OF TRAIN--#

    def test_rnn_model(self):
        '''
        Train Basic RNN Cell model, Recurrance within the main cell body
        '''
        tf.reset_default_graph()#Reset TensorFlow information

        #--TOP HEAVY VARIABLES--#
        dataset_x = self.test_data#Data
        dataset_y = self.test_label#Label
        steps = self.test_step#Step Number
        num_in = self.n_inputs#Number of Inputs
        num_out = self.n_outputs#Number of Outputs
        num_data = self.test_size

        #Embedded Batcher
        def next_batch():
            '''
            Internal Next Training Batcher for input data into network
            '''
            #Variables
            x_batch = []
            y_batch = []
            issue_warn = False

            #The lord have mercy upon this horrific code:
            #Checker
            temp_checker = self.current_pos + steps
            if temp_checker > self.test_size:
                #Issue
                issue_warn = True
            else:
                #No Issue
                issue_warn = False

            #Main loop
            if self.current_pos == 0:#If Started
                t_min = 0#Set Min
                t_max = t_min + self.n_steps#Set Max
                idx = np.arange(t_min,t_max)
                x_batch = [dataset_x[i] for i in idx]
                #Position Index in statements
                self.current_pos = self.current_pos * 0#Reset
                self.current_pos = self.current_pos + t_max#Set for next

            elif issue_warn == False and self.current_pos != 0:#Regular Batch
                t_min = self.current_pos#Set Min
                t_max = t_min + self.n_steps#Set Max
                idx = np.arange(t_min,t_max)
                x_batch = [dataset_x[i] for i in idx]

                #Position Index in statements
                self.current_pos = self.current_pos * 0#Reset
                self.current_pos = self.current_pos + t_max#Set for next

            elif issue_warn == True and self.current_pos != 0:#If T_Max over limit
                #t_min = self.t_backup OLD
                t_min = num_data - steps
                t_max = num_data#Set Maximum
                idx = np.arange(t_min,t_max)
                x_batch = [dataset_x[i] for i in idx]

            #Reshape
            x_batch = array(x_batch).reshape(1,steps,num_in)
            #Return Batching
            return x_batch

        X = tf.placeholder(tf.float32, shape=[1, steps, num_in], name="x")#INPUT_BATCH_DATA_PLACEHOLDER

        '''
        #Layer declare: Layer[(Number of cells, What Activation FN)Number of layers]
        cells = [
            tf.contrib.rnn.BasicRNNCell(num_units=self.n_neurons, activation=self.activation_function)
            for layer in range(self.n_layers)]
        '''

        #Declare Cell
        rnn_cell=[]
        for _ in range(self.n_layers):
            #Layer declare: Layer[(Number of cells, What Activation FN)Number of layers]
            cell = tf.contrib.rnn.BasicRNNCell(num_units=self.n_neurons, activation=self.activation_function)
            #Dropper layer for over saturated data
            rnn_cell.append(cell)

        #Stack cells to layers
        multi_layer_cell = tf.contrib.rnn.MultiRNNCell(rnn_cell)#Append cell to layers
        rnn_outputs, states = tf.nn.dynamic_rnn(multi_layer_cell, X, dtype=tf.float32)#Declare outputs and states for loss

        #OUTPUT PROJECT WRAPPER--For input Data
        stacked_rnn_outputs = tf.reshape(rnn_outputs, [-1, self.n_neurons])
        stacked_outputs = tf.layers.dense(stacked_rnn_outputs, num_out)
        outputs = tf.reshape(stacked_outputs, [steps, num_out])

        #Create Saver
        saver = tf.train.Saver()

        #--Main Session--#
        with tf.Session() as sess:
            saver.restore(sess, "Recurrent_Network/Saves/RNN/Recurrent_Neural_Network.ckpt")#Restore Files
            #---Outer Epoch Shell---#
            print("---Testing RNN Model---")
            for epoch  in range(1):
                #---Inner Iteration shell----#
                self.test_results = []*0#Zero 
                for iteration in range(self.num_batch):
                    #---Execution Shell---#
                    x_batch = next_batch()
                    output_val = sess.run([outputs], feed_dict={X: x_batch})
                    self.test_results.append(output_val)
                    #---Execution Shell End---#
                #---Inner Shell End---#
                print("Testing: Epoch: [",epoch ,"/" , self.epoch,"] ")#Prompter ,"Output: [", output_val, "]"", "MSE: [",mse,"]"
            #---Outer Shell End---#
            print("---Testing Completed!---")

            #File Printer
            filereader.read_into_temp(np.ravel(self.test_results), "/RNN/test_results.csv")#Print Into File

            #Numpy.Ravel
            training_results = np.ravel(self.train_results)
            testing_results = np.ravel(self.test_results)

            #Plotters
            print("Plotting...")
            dataplotter.test_graph(self.filename,testing_results, 'rnn')
            dataplotter.c_test_graph(self.filename,testing_results, np.ravel(self.test_label), 'rnn')
            dataplotter.c_combined_graph(self.filename, self.train_label, self.test_label,np.ravel(self.train_results), np.ravel(self.test_results), 'rnn')
            print("Plot Completed!")

            print("\n #---End of Recurrent Neural Network---# \n")
            #---END OF TEST--#
        #Return Dataset
        return self.test_results

    #--------------------------#
    #Bi-Directional Recurrent Neural Networks# #INCOMPLETE#

    def train_bi_rnn_model(self):
        '''
        Train Deep Bi-Directional RNN Cell model, Recurrance within the main cell body
        '''
        tf.reset_default_graph()#Reset

        #--TOP HEAVY VARIABLES--#
        dataset_x = self.train_data#Data
        dataset_y = self.train_label#Label
        steps = self.train_step#Step Number
        num_in = self.n_inputs#Number of Inputs
        num_out = self.n_outputs#Number of Outputs
        prob = 1.0#Keep Prop for Outputs of Neural Network
        num_data = self.train_size#Size of Network

        #Embedded Batcher
        def next_batch():
            '''
            Internal Next Training Batcher for input data into network
            '''
            #Local Variables
            x_batch = []
            y_batch = []
            issue_warn = False

            #The lord have mercy upon this horrific code:
            #Checker
            temp_checker = self.current_pos + steps
            if temp_checker > num_data:
                #Issue
                issue_warn = True
            else:
                #No Issue
                issue_warn = False

            #Main loop
            if self.current_pos == 0:#If Started
                t_min = 0#Set Min
                t_max = t_min + steps#Set Max
                idx = np.arange(t_min,t_max)
                x_batch = [dataset_x[i] for i in idx]
                y_batch = [dataset_y[i] for i in idx]
                #Position Index in statements
                self.current_pos = self.current_pos * 0#Reset
                self.current_pos = self.current_pos + t_max#Set for next

            elif issue_warn == False and self.current_pos != 0:#Regular Batch
                t_min = self.current_pos#Set Min
                t_max = t_min + steps#Set Max
                idx = np.arange(t_min,t_max)
                x_batch = [dataset_x[i] for i in idx]
                y_batch = [dataset_y[i] for i in idx]

                #Position Index in statements
                self.current_pos = self.current_pos * 0#Reset
                self.current_pos = self.current_pos + t_max#Set for next

                #OLD
                self.t_backup = self.t_backup * 0#Reset
                self.t_backup = self.current_pos#Setnew

            elif issue_warn == True and self.current_pos != 0:#If T_Max over limit
                #t_min = self.t_backup OLD
                t_min = num_data - steps
                t_max = num_data#Set Maximum
                idx = np.arange(t_min,t_max)
                x_batch = [dataset_x[i] for i in idx]
                y_batch = [dataset_y[i] for i in idx]

            #Reshape
            x_batch = array(x_batch).reshape(1,steps,num_in)
            y_batch = array(y_batch).reshape(steps,num_out)
            #Return Batching
            return x_batch, y_batch

        X = tf.placeholder(tf.float32, shape=[1, steps, num_in], name="x")#INPUT_BATCH_DATA_PLACEHOLDER Rank 3 Matrix
        Y = tf.placeholder(tf.float32, shape=[steps, num_out], name="y")#TARGET_BATCH_DATA_PLACEHOLDER

        '''#OLD
        #Layer declare: Layer[(Number of cells, What Activation FN)Number of layers]
        layers = [
            tf.contrib.rnn.BasicRNNCell(num_units=self.n_neurons, activation=self.activation_function)
            for layer in range(self.n_layers)]

        #Dropper layer for over saturated data
        cells = [tf.contrib.rnn.DropoutWrapper(cell,input_keep_prob=prob, output_keep_prob=prob, state_keep_prob=prob) for cell in layers]
        '''

        #Declare Cell
        rnn_cell=[]
        for _ in range(self.n_layers):
            #Layer declare: Layer[(Number of cells, What Activation FN)Number of layers]
            cell = tf.contrib.rnn.BasicRNNCell(num_units=self.n_neurons, activation=self.activation_function)
            #Dropper layer for over saturated data
            cell = tf.nn.rnn_cell.DropoutWrapper(cell,input_keep_prob=prob, output_keep_prob=prob, state_keep_prob=prob)
            rnn_cell.append(cell)

        #Stack cells to layers
        multi_layer_cell = tf.contrib.rnn.MultiRNNCell(rnn_cell)#Append cell to layers
        rnn_outputs, states = tf.nn.dynamic_rnn(multi_layer_cell, X, dtype=tf.float32)#Declare outputs and states for loss

        #OUTPUT PROJECT WRAPPER--For input Data
        with tf.name_scope("wrapper"):
            '''
            Output Projection Wrapper
            '''
            stacked_rnn_outputs = tf.reshape(rnn_outputs, [-1, self.n_neurons])
            stacked_outputs = tf.layers.dense(stacked_rnn_outputs, num_out)
            outputs = tf.reshape(stacked_outputs, [steps, num_out])

        #LOSS
        with tf.name_scope("loss"):
            loss = tf.reduce_sum(tf.square(outputs - Y))#Sum of Loss
            optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate)#OPTIMIZER, ADAM is MOST ACCURATE MODEL
            '''
            Adam Optimizer is an extenstion to Stochastic Gradient Descent. Designed by Diderik Kingma from OpenAI and Jimmy Ba from Tornoto from;
            “Adam: A Method for Stochastic Optimization“:https://arxiv.org/abs/1412.6980;

            Learning_Rate /or/ Alpha = Is the Porportion that weights are updated(0.0001 for smaller variables, 0.3 for larger)
            beta1 = Exponential decay rate for the first moment estimates
            beta2 = Exopential decay rate for second moment estimates
            epsilon = Number to prevent division by zero, DO NOT MODIFY
            '''
            training_op = optimizer.minimize(loss)

        #Initialize Variables
        init = tf.global_variables_initializer()
        #Create Saver
        saver = tf.train.Saver()

        #--Main Session--#
        with tf.Session() as sess:
            init.run()#Initialize
            #---Outer Epoch Shell---#
            print("---Training RNN Model---")
            for epoch  in range(self.epoch):
                #---Inner Iteration shell----#
                self.train_results = []*0#Zero Results to grab last epoch(Most Accurate)
                for iteration in range(self.num_batch):
                    #---Execution Shell---#
                    x_batch, y_batch = next_batch()#Train           
                    _, output_val, mse = sess.run([training_op, outputs, loss], feed_dict={X: x_batch, Y: y_batch})#TRAINING
                    results_data = []#Append data to array
                    for i in range(len(output_val)):
                        results_data.append(output_val[i])
                    self.train_results.append(results_data)#Append Results
                    #---Execution Shell End---#
                #---Inner Shell End---#
                print("Training RNN: Epoch: [",epoch ,"/" , self.epoch,"] ", "MSE: [",mse,"]")#Prompter ,"Output: [", output_val, "]"
                error = str(mse)
                self.error_loss.append(error)
            #---Outer Shell End---#
            print("---Training Completed!---")
            print("Saving!")
            saver.save(sess, "Recurrent_Network/Saves/RNN/Recurrent_Neural_Network.ckpt")
            print("Printing to Files")
            filereader.read_into_temp(np.ravel(self.train_results), "/RNN/train_results.csv")#Save to file
            filereader.read_into_temp(self.error_loss, "/RNN/error_results.csv")#Save to File
            print("Task Completed!")
            #Plotters
            print("Plotting...")
            dataplotter.train_graph(self.filename,np.ravel(self.train_results), 'rnn')
            dataplotter.plt_error_graph(self.filename, np.ravel(self.error_loss), self.epoch, 'rnn')
            dataplotter.c_train_graph(self.filename,np.ravel(self.train_results), np.ravel(self.train_label), 'rnn')
            print("Plot Completed!")
            #---END OF TRAIN--#
        #Return Dataset
        return self.train_results

    def test_bi_rnn_model(self):
        '''
        Train Basic RNN Cell model, Recurrance within the main cell body
        '''
        tf.reset_default_graph()#Reset

        #--TOP HEAVY VARIABLES--#
        dataset_x = self.test_data#Data
        dataset_y = self.test_label#Label
        steps = self.test_step#Step Number
        num_in = self.n_inputs#Number of Inputs
        num_out = self.n_outputs#Number of Outputs
        num_data = self.test_size

        #Embedded Batcher
        def next_batch():
            '''
            Internal Next Training Batcher for input data into network
            '''
            #Variables
            x_batch = []
            y_batch = []
            issue_warn = False

            #The lord have mercy upon this horrific code:
            #Checker
            temp_checker = self.current_pos + steps
            if temp_checker > self.test_size:
                #Issue
                issue_warn = True
            else:
                #No Issue
                issue_warn = False

            #Main loop
            if self.current_pos == 0:#If Started
                t_min = 0#Set Min
                t_max = t_min + self.n_steps#Set Max
                idx = np.arange(t_min,t_max)
                x_batch = [dataset_x[i] for i in idx]
                #Position Index in statements
                self.current_pos = self.current_pos * 0#Reset
                self.current_pos = self.current_pos + t_max#Set for next

            elif issue_warn == False and self.current_pos != 0:#Regular Batch
                t_min = self.current_pos#Set Min
                t_max = t_min + self.n_steps#Set Max
                idx = np.arange(t_min,t_max)
                x_batch = [dataset_x[i] for i in idx]

                #Position Index in statements
                self.current_pos = self.current_pos * 0#Reset
                self.current_pos = self.current_pos + t_max#Set for next

            elif issue_warn == True and self.current_pos != 0:#If T_Max over limit
                #t_min = self.t_backup OLD
                t_min = num_data - steps
                t_max = num_data#Set Maximum
                idx = np.arange(t_min,t_max)
                x_batch = [dataset_x[i] for i in idx]

            #Reshape
            x_batch = array(x_batch).reshape(1,steps,num_in)
            #Return Batching
            return x_batch

        X = tf.placeholder(tf.float32, shape=[1, steps, num_in], name="x")#INPUT_BATCH_DATA_PLACEHOLDER

        '''
        #Layer declare: Layer[(Number of cells, What Activation FN)Number of layers]
        cells = [
            tf.contrib.rnn.BasicRNNCell(num_units=self.n_neurons, activation=self.activation_function)
            for layer in range(self.n_layers)]
        '''

        #Declare Cell
        rnn_cell=[]
        for _ in range(self.n_layers):
            #Layer declare: Layer[(Number of cells, What Activation FN)Number of layers]
            cell = tf.contrib.rnn.BasicRNNCell(num_units=self.n_neurons, activation=self.activation_function)
            #Dropper layer for over saturated data
            rnn_cell.append(cell)

        #Stack cells to layers
        multi_layer_cell = tf.contrib.rnn.MultiRNNCell(rnn_cell)#Append cell to layers
        rnn_outputs, states = tf.nn.dynamic_rnn(multi_layer_cell, X, dtype=tf.float32)#Declare outputs and states for loss

        #OUTPUT PROJECT WRAPPER--For input Data
        stacked_rnn_outputs = tf.reshape(rnn_outputs, [-1, self.n_neurons])
        stacked_outputs = tf.layers.dense(stacked_rnn_outputs, num_out)
        outputs = tf.reshape(stacked_outputs, [steps, num_out])

        #Create Saver
        saver = tf.train.Saver()

        #--Main Session--#
        with tf.Session() as sess:
            saver.restore(sess, "Recurrent_Network/Saves/RNN/Recurrent_Neural_Network.ckpt")#Restore Files
            #---Outer Epoch Shell---#
            print("---Testing RNN Model---")
            for epoch  in range(1):
                #---Inner Iteration shell----#
                self.test_results = []*0#Zero 
                for iteration in range(self.num_batch):
                    #---Execution Shell---#
                    x_batch = next_batch()
                    output_val = sess.run([outputs], feed_dict={X: x_batch})
                    self.test_results.append(output_val)
                    #---Execution Shell End---#
                #---Inner Shell End---#
                print("Testing: Epoch: [",epoch ,"/" , self.epoch,"] ")#Prompter ,"Output: [", output_val, "]"", "MSE: [",mse,"]"
            #---Outer Shell End---#
            print("---Testing Completed!---")
            filereader.read_into_temp(np.ravel(self.test_results), "/RNN/test_results.csv")#Print Into File

            #Numpy.Ravel
            training_results = np.ravel(self.train_results)
            testing_results = np.ravel(self.test_results)

            #Plotters
            print("Plotting...")
            dataplotter.test_graph(self.filename,testing_results, 'rnn')
            dataplotter.c_test_graph(self.filename,testing_results, np.ravel(self.test_label), 'rnn')
            dataplotter.c_combined_graph(self.filename, self.train_label, self.test_label,np.ravel(self.train_results), np.ravel(self.test_results), 'rnn')
            print("Plot Completed!")

            self.rnn_data = self.test_results
            print("\n #---End of Recurrent Neural Network---# \n")
            #---END OF TEST--#
        #Return Dataset
        return self.test_results

    #--------------------------# #UPDATE#
    #Long Short-Term Memory#

    def train_lstm_model(self):
        '''
        Train Deep LSTM Cell model, Recurrance within the main cell body
        '''
        tf.reset_default_graph()#Reset
        self.error_loss = []*0#Reset Error

        #--TOP HEAVY VARIABLES--#
        dataset_x = self.train_data#Data
        dataset_y = self.train_label#Label
        steps = self.train_step#Step Number
        num_in = self.n_inputs#Number of Inputs
        num_out = self.n_outputs#Number of Outputs
        prob = 1.0#Keep Prop for Outputs of Neural Network
        num_data = self.train_size#Size of Network

        #Embedded Batcher
        def next_batch():
            '''
            Internal Next Training Batcher for input data into network
            '''
            #Local Variables
            x_batch = []
            y_batch = []
            issue_warn = False

            #The lord have mercy upon this horrific code:
            #Checker
            temp_checker = self.current_pos + steps
            if temp_checker > num_data:
                #Issue
                issue_warn = True
            else:
                #No Issue
                issue_warn = False

            #Main loop
            if self.current_pos == 0:#If Started
                t_min = 0#Set Min
                t_max = t_min + steps#Set Max
                idx = np.arange(t_min,t_max)
                x_batch = [dataset_x[i] for i in idx]
                y_batch = [dataset_y[i] for i in idx]
                #Position Index in statements
                self.current_pos = self.current_pos * 0#Reset
                self.current_pos = self.current_pos + t_max#Set for next

            elif issue_warn == False and self.current_pos != 0:#Regular Batch
                t_min = self.current_pos#Set Min
                t_max = t_min + steps#Set Max
                idx = np.arange(t_min,t_max)
                x_batch = [dataset_x[i] for i in idx]
                y_batch = [dataset_y[i] for i in idx]

                #Position Index in statements
                self.current_pos = self.current_pos * 0#Reset
                self.current_pos = self.current_pos + t_max#Set for next

                #OLD
                self.t_backup = self.t_backup * 0#Reset
                self.t_backup = self.current_pos#Setnew

            elif issue_warn == True and self.current_pos != 0:#If T_Max over limit
                #t_min = self.t_backup OLD
                t_min = num_data - steps
                t_max = num_data#Set Maximum
                idx = np.arange(t_min,t_max)
                x_batch = [dataset_x[i] for i in idx]
                y_batch = [dataset_y[i] for i in idx]

            #Reshape
            x_batch = array(x_batch).reshape(1,steps,num_in)
            y_batch = array(y_batch).reshape(steps,num_out)
            #Return Batching
            return x_batch, y_batch

        X = tf.placeholder(tf.float32, shape=[1, steps, num_in], name="x")#INPUT_BATCH_DATA_PLACEHOLDER
        Y = tf.placeholder(tf.float32, shape=[steps, num_out], name="y")#TARGET_BATCH_DATA_PLACEHOLDER
        
        '''
        #LSTM
        layers = [
            tf.contrib.rnn.BasicLSTMCell(num_units=self.n_neurons,forget_bias=1.0, activation=self.activation_function)
            for layer in range(self.n_layers)]

        #Dropper layer for over saturated data
        cells = [tf.contrib.rnn.DropoutWrapper(cell,input_keep_prob=prob, output_keep_prob=prob, state_keep_prob=prob) for cell in layers]
        '''

        #Main LSTM Framework#
        lstm_cell=[]
        for _ in range(self.n_layers):
            cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.n_neurons, forget_bias=1.0, activation=self.activation_function)#Layer declare: Layer[(Number of cells, What Activation FN)Number of layers]
            cell = tf.nn.rnn_cell.DropoutWrapper(cell,input_keep_prob=prob, output_keep_prob=prob, state_keep_prob=prob)#Dropper layer for over saturated data
            lstm_cell.append(cell)#Append to rnn_cell

        #Stack cells to layers
        multi_layer_cell = tf.contrib.rnn.MultiRNNCell(lstm_cell)
        rnn_outputs, states = tf.nn.dynamic_rnn(multi_layer_cell, X, dtype=tf.float32)

        #OUTPUT PROJECT WRAPPER--For input Data
        with tf.name_scope("wrapper"):
            stacked_rnn_outputs = tf.reshape(rnn_outputs, [-1, self.n_neurons])
            stacked_outputs = tf.layers.dense(stacked_rnn_outputs, num_out)
            outputs = tf.reshape(stacked_outputs, [steps, num_out])

        #LOSS
        with tf.name_scope("loss"):
            loss = tf.reduce_sum(tf.square(outputs - Y))#Sum of Loss
            optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate)#OPTIMIZER, ADAM is MOST ACCURATE MODEL
            training_op = optimizer.minimize(loss)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        #--Main Session--#
        with tf.Session() as sess:
            init.run()#Initialize
            #---Outer Epoch Shell---#
            print("---Training LSTM Model---")
            for epoch  in range(self.epoch):
                #---Inner Iteration shell----#
                self.train_results = []*0#Zero Results to grab last epoch(Most Accurate)
                for iteration in range(self.num_batch):
                    #---Execution Shell---#
                    x_batch, y_batch = next_batch()#Train           
                    _, output_val, mse = sess.run([training_op, outputs, loss], feed_dict={X: x_batch, Y: y_batch})#TRAINING
                    results_data = []#Append data to array
                    for i in range(len(output_val)):
                        results_data.append(output_val[i])
                    self.train_results.append(results_data)#Append Results
                    #---Execution Shell End---#
                #---Inner Shell End---#
                print("Training LSTM: Epoch: [",epoch ,"/" , self.epoch,"] ", "MSE: [",mse,"]")#Prompter ,"Output: [", output_val, "]"
                error = str(mse)
                self.error_loss.append(error)
            #---Outer Shell End---#
            print("---Training Completed!---")
            print("Saving!")
            saver.save(sess, "Recurrent_Network/Saves/LSTM/LSTM_Network.ckpt")
            print("Printing to Files")
            filereader.read_into_temp(np.ravel(self.train_results), "/LSTM/train_results.csv")#Save to file
            filereader.read_into_temp(self.error_loss, "/LSTM/error_results.csv")#Save to File
            print("Task Completed!")
            #Plotters
            print("Plotting...")
            dataplotter.train_graph(self.filename,np.ravel(self.train_results), 'lstm')
            dataplotter.plt_error_graph(self.filename, np.ravel(self.error_loss), self.epoch, 'lstm')
            dataplotter.c_train_graph(self.filename,np.ravel(self.train_results), np.ravel(self.train_label), 'lstm')
            print("Plot Completed!")
            #---END OF TRAIN--#

        #Return Dataset
        return self.train_results

    def test_lstm_model(self):
        '''
        Train Basic RNN Cell model, Recurrance within the main cell body
        '''
        tf.reset_default_graph()#Reset

        #--TOP HEAVY VARIABLES--#
        dataset_x = self.test_data#Data
        dataset_y = self.test_label#Label
        steps = self.test_step#Step Number
        num_in = self.n_inputs#Number of Inputs
        num_out = self.n_outputs#Number of Outputs
        num_data = self.test_size

        #Embedded Batcher
        def next_batch():
            '''
            Internal Next Training Batcher for input data into network
            '''
            #Variables
            x_batch = []
            y_batch = []
            issue_warn = False

            #The lord have mercy upon this horrific code:
            #Checker
            temp_checker = self.current_pos + steps
            if temp_checker > self.test_size:
                #Issue
                issue_warn = True
            else:
                #No Issue
                issue_warn = False

            #Main loop
            if self.current_pos == 0:#If Started
                t_min = 0#Set Min
                t_max = t_min + self.n_steps#Set Max
                idx = np.arange(t_min,t_max)
                x_batch = [dataset_x[i] for i in idx]
                #Position Index in statements
                self.current_pos = self.current_pos * 0#Reset
                self.current_pos = self.current_pos + t_max#Set for next

            elif issue_warn == False and self.current_pos != 0:#Regular Batch
                t_min = self.current_pos#Set Min
                t_max = t_min + self.n_steps#Set Max
                idx = np.arange(t_min,t_max)
                x_batch = [dataset_x[i] for i in idx]

                #Position Index in statements
                self.current_pos = self.current_pos * 0#Reset
                self.current_pos = self.current_pos + t_max#Set for next

            elif issue_warn == True and self.current_pos != 0:#If T_Max over limit
                #t_min = self.t_backup OLD
                t_min = num_data - steps
                t_max = num_data#Set Maximum
                idx = np.arange(t_min,t_max)
                x_batch = [dataset_x[i] for i in idx]

            #Reshape
            x_batch = array(x_batch).reshape(1,steps,num_in)
            #Return Batching
            return x_batch

        X = tf.placeholder(tf.float32, shape=[1, steps, num_in], name="x")#INPUT_BATCH_DATA_PLACEHOLDER

        '''
        #Layer declare: Layer[(Number of cells, What Activation FN)Number of layers]
        cells = [
            tf.contrib.rnn.BasicLSTMCell(num_units=self.n_neurons, activation=self.activation_function)
            for layer in range(self.n_layers)]
        '''

        #Main LSTM Framework#
        lstm_cell=[]
        for _ in range(self.n_layers):
            cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.n_neurons, forget_bias=1.0, activation=self.activation_function)#Layer declare: Layer[(Number of cells, What Activation FN)Number of layers]
            rnn_cell.append(cell)#Append to rnn_cell

        #Stack cells to layers
        multi_layer_cell = tf.contrib.rnn.MultiRNNCell(rnn_cell)#Append cell to layers
        rnn_outputs, states = tf.nn.dynamic_rnn(multi_layer_cell, X, dtype=tf.float32)#Declare outputs and states for loss

        #OUTPUT PROJECT WRAPPER--For input Data
        stacked_rnn_outputs = tf.reshape(rnn_outputs, [-1, self.n_neurons])
        stacked_outputs = tf.layers.dense(stacked_rnn_outputs, num_out)
        outputs = tf.reshape(stacked_outputs, [steps, num_out])

        #Create Saver
        saver = tf.train.Saver()

        #--Main Session--#
        with tf.Session() as sess:
            saver.restore(sess, "Recurrent_Network/Saves/RNN/Recurrent_Neural_Network.ckpt")#Restore Files
            #---Outer Epoch Shell---#
            print("---Testing LSTM Model---")
            for epoch  in range(1):
                #---Inner Iteration shell----#
                self.test_results = []*0#Zero 
                for iteration in range(self.num_batch):
                    #---Execution Shell---#
                    x_batch = next_batch()
                    output_val = sess.run([outputs], feed_dict={X: x_batch})
                    self.test_results.append(output_val)
                    #---Execution Shell End---#
                #---Inner Shell End---#
                print("Testing LSTM: Epoch: [",epoch ,"/" , self.epoch,"] ")#Prompter ,"Output: [", output_val, "]"", "MSE: [",mse,"]"
            #---Outer Shell End---#
            print("---Testing Completed!---")
            filereader.read_into_temp(self.test_results, "/LSTM/test_results.csv")#Print Into File

            #Numpy.Ravel
            training_results = np.ravel(self.train_results)
            testing_results = np.ravel(self.test_results)

            #Plotters
            print("Plotting...")
            dataplotter.test_graph(self.filename,testing_results, 'lstm')
            dataplotter.c_test_graph(self.filename,testing_results, np.ravel(self.test_label), 'lstm')
            dataplotter.c_combined_graph(self.filename, self.train_label, self.test_label,np.ravel(self.train_results), np.ravel(self.test_results), 'lstm')
            #Plot Completed
            #dataplotter.co
            print("Plot Completed!")

            print("---LSTM END---")
            #---END OF TEST--#
        #Return Dataset
        return self.test_results

    #--------------------------#
    #Bi-Directional Long Short-Term Memory# #INCOMPLETE#

    def train_bi_lstm_model(self):
        '''
        Train Deep Bi-Directional LSTM Cell model, Recurrance within the main cell body
        '''
        tf.reset_default_graph()#Reset
        self.error_loss = []*0#Reset Error

        #--TOP HEAVY VARIABLES--#
        dataset_x = self.train_data#Data
        dataset_y = self.train_label#Label
        steps = self.train_step#Step Number
        num_in = self.n_inputs#Number of Inputs
        num_out = self.n_outputs#Number of Outputs
        prob = 1.0#Keep Prop for Outputs of Neural Network
        num_data = self.train_size#Size of Network

        #Embedded Batcher
        def next_batch():
            '''
            Internal Next Training Batcher for input data into network
            '''
            #Local Variables
            x_batch = []
            y_batch = []
            issue_warn = False

            #The lord have mercy upon this horrific code:
            #Checker
            temp_checker = self.current_pos + steps
            if temp_checker > num_data:
                #Issue
                issue_warn = True
            else:
                #No Issue
                issue_warn = False

            #Main loop
            if self.current_pos == 0:#If Started
                t_min = 0#Set Min
                t_max = t_min + steps#Set Max
                idx = np.arange(t_min,t_max)
                x_batch = [dataset_x[i] for i in idx]
                y_batch = [dataset_y[i] for i in idx]
                #Position Index in statements
                self.current_pos = self.current_pos * 0#Reset
                self.current_pos = self.current_pos + t_max#Set for next

            elif issue_warn == False and self.current_pos != 0:#Regular Batch
                t_min = self.current_pos#Set Min
                t_max = t_min + steps#Set Max
                idx = np.arange(t_min,t_max)
                x_batch = [dataset_x[i] for i in idx]
                y_batch = [dataset_y[i] for i in idx]

                #Position Index in statements
                self.current_pos = self.current_pos * 0#Reset
                self.current_pos = self.current_pos + t_max#Set for next

                #OLD
                self.t_backup = self.t_backup * 0#Reset
                self.t_backup = self.current_pos#Setnew

            elif issue_warn == True and self.current_pos != 0:#If T_Max over limit
                #t_min = self.t_backup OLD
                t_min = num_data - steps
                t_max = num_data#Set Maximum
                idx = np.arange(t_min,t_max)
                x_batch = [dataset_x[i] for i in idx]
                y_batch = [dataset_y[i] for i in idx]

            #Reshape
            x_batch = array(x_batch).reshape(1,steps,num_in)
            y_batch = array(y_batch).reshape(steps,num_out)
            #Return Batching
            return x_batch, y_batch

        X = tf.placeholder(tf.float32, shape=[1, steps, num_in], name="x")#INPUT_BATCH_DATA_PLACEHOLDER
        Y = tf.placeholder(tf.float32, shape=[steps, num_out], name="y")#TARGET_BATCH_DATA_PLACEHOLDER
        
        '''
        #LSTM
        layers = [
            tf.contrib.rnn.BasicLSTMCell(num_units=self.n_neurons,forget_bias=1.0, activation=self.activation_function)
            for layer in range(self.n_layers)]

        #Dropper layer for over saturated data
        cells = [tf.contrib.rnn.DropoutWrapper(cell,input_keep_prob=prob, output_keep_prob=prob, state_keep_prob=prob) for cell in layers]

        #Stack cells to layers
        multi_layer_cell = tf.contrib.rnn.MultiRNNCell(cells)
        rnn_outputs, states = tf.nn.dynamic_rnn(multi_layer_cell, X, dtype=tf.float32)
        '''

        fw_lstms = []
        for _ in range(self.n_layers):
            cell = tf.contrib.rnn.BasicLSTMCell(self.n_neurons) 
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=prob, output_keep_prob=prob, state_keep_prob=prob)
            fw_lstms.append(cell)

        bw_lstms = []
        for _ in range(self.n_layers):
            cell = tf.contrib.rnn.BasicLSTMCell(self.n_neurons) 
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=prob, output_keep_prob=prob, state_keep_prob=prob)
            bw_lstms.append(cell)    
        #bw_lstms = [tf.contrib.rnn.BasicLSTMCell(hidden_size) for _ in range(num_layers)]

        fw_init_state_ls = [lstm.zero_state(self.train_size, tf.float32) for lstm in fw_lstms]
        bw_init_state_ls = [lstm.zero_state(self.train_size, tf.float32) for lstm in bw_lstms]

        (fw_outputs, bw_outputs), final_states_fw, final_states_bw = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(cells_fw = fw_lstms, cells_bw = bw_lstms,
            inputs = X, dtype=tf.float32)

        #bi_final_state = tf.concat([final_states_fw[-1][1], final_states_bw[-1][1]], 1)
        
        with tf.name_scope("total_wrapper"):
            stacked_rnn_outputs = tf.reshape(rnn_outputs, [-1, self.n_neurons])
            stacked_outputs = tf.layers.dense(stacked_rnn_outputs, num_out)
            outputs = tf.reshape(stacked_outputs, [steps*2, num_out])
        '''
        with tf.name_scope("fw_wrapper"):
            stacked_rnn_outputs = tf.reshape(fw_outputs, [-1, self.n_neurons])
            stacked_outputs = tf.layers.dense(stacked_rnn_outputs, num_out)
            fw_outputs = tf.reshape(stacked_outputs, [steps, num_out])

        with tf.name_scope("bw_wrapper"):
            stacked_rnn_outputs = tf.reshape(bw_outputs, [-1, self.n_neurons])
            stacked_outputs = tf.layers.dense(stacked_rnn_outputs, num_out)
            bw_outputs = tf.reshape(stacked_outputs, [steps, num_out])
        '''

        bi_final_state = tf.concat([final_states_fw[-1][1], final_states_bw[-1][1]], 1)

        #LOSS
        with tf.name_scope("loss"):
            loss = tf.reduce_sum(tf.square(rnn_outputs - Y))#Sum of Loss
            optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate)#OPTIMIZER, ADAM is MOST ACCURATE MODEL
            training_op = optimizer.minimize(loss)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        #--Main Session--#
        with tf.Session() as sess:
            init.run()#Initialize
            #---Outer Epoch Shell---#
            print("---Training LSTM Model---")
            for epoch  in range(self.epoch):
                #---Inner Iteration shell----#
                self.train_results = []*0#Zero Results to grab last epoch(Most Accurate)
                for iteration in range(self.num_batch):
                    #---Execution Shell---#
                    x_batch, y_batch = next_batch()#Train           
                    _, output_val, mse = sess.run([training_op, outputs, loss], feed_dict={X: x_batch, Y: y_batch})#TRAINING
                    #_, fw_results, bw_results, mse = sess.run([training_op, fw_outputs, bw, loss], feed_dict={X: x_batch, Y: y_batch})#TRAINING
                    results_data = []#Append data to array
   
                    #Split Data in half
                    first_half = output_val[:len(output_val/2)]#First Half
                    first_half = output_val[len(output_val/2):]#Second Half

                    #Append Forward
                    for i in range(len(first_half)):
                        results_data.append(first_half[i])
                    self.train_results.append(results_data)#Append Results

                    #Append Backward
                    for i in range(len(first_half)):
                        '''
                        '''

                    #Append Backward
                    
                    #---Execution Shell End---#
                #---Inner Shell End---#
                print("Training LSTM: Epoch: [",epoch ,"/" , self.epoch,"] ", "MSE: [",mse,"]")#Prompter ,"Output: [", output_val, "]"
                error = str(mse)
                self.error_loss.append(error)
            #---Outer Shell End---#
            print("---Training Completed!---")

            #Saving
            print("Saving!")
            saver.save(sess, "Recurrent_Network/Saves/RNN/Recurrent_Neural_Network.ckpt")

            #Printers
            print("Printing to Files")
            filereader.read_into_temp(np.ravel(self.train_results), "/BI-LSTM/train_results.csv")#Save to file
            filereader.read_into_temp(self.error_loss, "/BI-LSTM/error_results.csv")#Save to File
            print("Task Completed!")

            #Plotters
            print("Plotting...")
            dataplotter.train_graph(self.filename,np.ravel(self.train_results), 'bi-lstm')
            dataplotter.plt_error_graph(self.filename, np.ravel(self.error_loss), self.epoch, 'bi-lstm')
            dataplotter.c_train_graph(self.filename,np.ravel(self.train_results), np.ravel(self.train_label), 'bi-lstm')
            print("Plot Completed!")
            #---END OF TRAIN--#

        #Return Dataset
        return self.train_results

    def test_bi_lstm_model(self):
        '''
        Train Basic RNN Cell model, Recurrance within the main cell body
        '''
        tf.reset_default_graph()#Reset

        #--TOP HEAVY VARIABLES--#
        dataset_x = self.test_data#Data
        dataset_y = self.test_label#Label
        steps = self.test_step#Step Number
        num_in = self.n_inputs#Number of Inputs
        num_out = self.n_outputs#Number of Outputs
        num_data = self.test_size

        #Embedded Batcher
        def next_batch():
            '''
            Internal Next Training Batcher for input data into network
            '''
            #Variables
            x_batch = []
            y_batch = []
            issue_warn = False

            #The lord have mercy upon this horrific code:
            #Checker
            temp_checker = self.current_pos + steps
            if temp_checker > self.test_size:
                #Issue
                issue_warn = True
            else:
                #No Issue
                issue_warn = False

            #Main loop
            if self.current_pos == 0:#If Started
                t_min = 0#Set Min
                t_max = t_min + self.n_steps#Set Max
                idx = np.arange(t_min,t_max)
                x_batch = [dataset_x[i] for i in idx]
                #Position Index in statements
                self.current_pos = self.current_pos * 0#Reset
                self.current_pos = self.current_pos + t_max#Set for next

            elif issue_warn == False and self.current_pos != 0:#Regular Batch
                t_min = self.current_pos#Set Min
                t_max = t_min + self.n_steps#Set Max
                idx = np.arange(t_min,t_max)
                x_batch = [dataset_x[i] for i in idx]

                #Position Index in statements
                self.current_pos = self.current_pos * 0#Reset
                self.current_pos = self.current_pos + t_max#Set for next

            elif issue_warn == True and self.current_pos != 0:#If T_Max over limit
                #t_min = self.t_backup OLD
                t_min = num_data - steps
                t_max = num_data#Set Maximum
                idx = np.arange(t_min,t_max)
                x_batch = [dataset_x[i] for i in idx]

            #Reshape
            x_batch = array(x_batch).reshape(1,steps,num_in)
            #Return Batching
            return x_batch

        X = tf.placeholder(tf.float32, shape=[1, steps, num_in], name="x")#INPUT_BATCH_DATA_PLACEHOLDER

        #Layer declare: Layer[(Number of cells, What Activation FN)Number of layers]
        cells = [
            tf.contrib.rnn.BasicLSTMCell(num_units=self.n_neurons, activation=self.activation_function)
            for layer in range(self.n_layers)]

        #Stack cells to layers
        multi_layer_cell = tf.contrib.rnn.MultiRNNCell(cells)#Append cell to layers
        rnn_outputs, states = tf.nn.dynamic_rnn(multi_layer_cell, X, dtype=tf.float32)#Declare outputs and states for loss

        #OUTPUT PROJECT WRAPPER--For input Data
        stacked_rnn_outputs = tf.reshape(rnn_outputs, [-1, self.n_neurons])
        stacked_outputs = tf.layers.dense(stacked_rnn_outputs, num_out)
        outputs = tf.reshape(stacked_outputs, [steps, num_out])

        #Create Saver
        saver = tf.train.Saver()

        #--Main Session--#
        with tf.Session() as sess:
            saver.restore(sess, "Recurrent_Network/Saves/RNN/Recurrent_Neural_Network.ckpt")#Restore Files
            #---Outer Epoch Shell---#
            print("---Testing LSTM Model---")
            for epoch  in range(1):
                #---Inner Iteration shell----#
                self.test_results = []*0#Zero 
                for iteration in range(self.num_batch):
                    #---Execution Shell---#
                    x_batch = next_batch()
                    output_val = sess.run([outputs], feed_dict={X: x_batch})
                    self.test_results.append(output_val)
                    #---Execution Shell End---#
                #---Inner Shell End---#
                print("Testing LSTM: Epoch: [",epoch ,"/" , self.epoch,"] ")#Prompter ,"Output: [", output_val, "]"", "MSE: [",mse,"]"
            #---Outer Shell End---#
            print("---Testing Completed!---")
            filereader.read_into_temp(self.test_results, "/LSTM/test_results.csv")#Print Into File

            #Numpy.Ravel
            training_results = np.ravel(self.train_results)
            testing_results = np.ravel(self.test_results)

            #Plotters
            print("Plotting...")
            dataplotter.test_graph(self.filename,testing_results, 'lstm')
            dataplotter.c_test_graph(self.filename,testing_results, np.ravel(self.test_label), 'lstm')
            dataplotter.c_combined_graph(self.filename, self.train_label, self.test_label,np.ravel(self.train_results), np.ravel(self.test_results), 'lstm')
            #Plot Completed
            #dataplotter.co
            print("Plot Completed!")

            print("---LSTM END---")
            #---END OF TEST--#
        #Return Dataset
        return self.test_results

    #--------------------------#

    def plot_concat_graphs(self):
        '''
        Plots and stores nn_error, nn_train, nn_test graphs 
        '''