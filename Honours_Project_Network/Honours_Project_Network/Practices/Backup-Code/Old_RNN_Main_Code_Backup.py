    def basic_rnn_model(self, training=bool):
        '''
        Train Recurrent Neural Network model from input data and parameters
        '''
        #Prompt Parameters prior to injection
        print("Initalize Train RNN: ", "\n","STRUCTURE:", "EACH LAYER has [", self.n_neurons, "] and there are [", self.n_layers,"] LAYERS \n")
        print("PARAMETERS: ", "LEARNING RATE start set [", self.learning_rate, "]")
        print("DATA: ", "FILENAME as [", self.filename,"] With [", self.num_data,"] VARIABLES in ARRAY")

        #Train Graph
        train = tf.Graph()
        test = tf.Graph()

        with train.as_default():
            cells = []

        '''
        #Layer declare: Layer[(Number of cells, What Activation FN)Number of layers]
        cells = [
            tf.contrib.rnn.BasicRNNCell(num_units=self.n_neurons, activation=self.activation_function)
            for layer in range(self.n_layers)]

        if training:
            #Dropper layer for over saturated data
            cells = [tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=1.0) for cell in cells]

        #Stack cells to layers
        multi_layer_cell = tf.contrib.rnn.MultiRNNCell(cells)#Append cell to layers
        rnn_outputs, states = tf.nn.dynamic_rnn(multi_layer_cell, self.X, dtype=tf.float32)#Declare outputs and states for loss

        #OUTPUT PROJECT WRAPPER--For input Data
        stacked_rnn_outputs = tf.reshape(rnn_outputs, [-1, self.n_neurons])
        stacked_outputs = tf.layers.dense(stacked_rnn_outputs, self.n_outputs)
        outputs = tf.reshape(stacked_outputs, [self.n_steps, self.n_outputs])

        #LOSS
        loss = tf.reduce_sum(tf.square(outputs - self.Y))
        optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
        training_op = optimizer.minimize(loss)

        #Initialize Variables
        init = tf.global_variables_initializer()
        #Create Saver
        saver = tf.train.Saver()
        '''
        #Main Session
        if training:
            '''
            Train Model with Dropout Wrapper
            '''
            print("---TRAINING---")

            #--Main Model--#
            layers = [
                tf.contrib.rnn.BasicRNNCell(num_units=self.n_neurons, activation=self.activation_function)
                for layer in range(self.n_layers)
                ]
            #Dropper layer for over saturated data
            cells = [tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=1.0, output_keep_prob=1.0) for cell in layers]

            #Stack cells to layers
            multi_layer_cell = tf.contrib.rnn.MultiRNNCell(cells)#Append cell to layers
            rnn_outputs, states = tf.nn.dynamic_rnn(multi_layer_cell, self.X, dtype=tf.float32)#Declare outputs and states for loss

            #OUTPUT PROJECT WRAPPER--For input Data
            stacked_rnn_outputs = tf.reshape(rnn_outputs, [-1, self.n_neurons])
            stacked_outputs = tf.layers.dense(stacked_rnn_outputs, self.n_outputs)
            outputs = tf.reshape(stacked_outputs, [self.n_steps, self.n_outputs])

            #LOSS
            loss = tf.reduce_sum(tf.square(outputs - self.Y))
            optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
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
                    self.results = []*0#Zero Results to grab last epoch(Most Accurate)
                    for iteration in range(self.num_batch):
                        #---Execution Shell---#
                        x_batch, y_batch = rnn.train_next_batch()#Train Inject
                        _, output_val, mse = sess.run([training_op, outputs, loss], feed_dict={self.X: x_batch, self.Y: y_batch})
                        #Append data to array
                        fucking_work = []
                        for i in range(len(output_val)):
                            fucking_work.append(output_val[i])
                        self.results.append(fucking_work)#Append Results
                        self.loss.append(mse)#Append Loss
                        #---Execution Shell End---#
                    #---Inner Shell End---#
                    print("Epoch: [",epoch ,"/" , self.epoch,"] ", "MSE: [",mse,"]")#Prompter ,"Output: [", output_val, "]"
                #---Outer Shell End---#
                print("---Training Completed!---")
                print("Saving!")
                saver.save(sess, "Recurrent_Network/Saves/my_honours_project_saver.ckpt")
                filereader.read_into_temp(self.results)

                #Plotters
                insert = np.ravel(self.results)
                print("Plotting...")
                dataplotter.train_graph(insert)
                dataplotter.c_train_graph(insert, self.input_data)
                print("Plot Completed!")
                #---END OF TRAIN--#
                print("Reseting!")
                tf.reset_default_graph#Reset All
                print("RESET!")

        else:
            '''
            Test Model without Dropout Wrapper
            '''
            print("---TESTING---")

            #--Main Model--#
            cell = [
                tf.contrib.rnn.BasicRNNCell(num_units=self.n_neurons, activation=self.activation_function)
                for layer in range(self.n_layers)
                ]

            #Stack cells to layers
            train_multi_layer_cell = tf.contrib.rnn.MultiRNNCell(cell)#Append cell to layers
            train_outputs, train_states = tf.nn.dynamic_rnn(train_multi_layer_cell, self.A, dtype=tf.float32)#Declare outputs and states for loss

            #OUTPUT PROJECT WRAPPER--For input Data
            stacked_rnn_outputs = tf.reshape(train_outputs, [-1, self.n_neurons])
            stacked_outputs = tf.layers.dense(stacked_rnn_outputs, self.n_outputs)
            outputs = tf.reshape(stacked_outputs, [self.n_steps, self.n_outputs])

            #LOSS
            loss = tf.reduce_sum(tf.square(outputs - self.B))
            optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
            training_op = optimizer.minimize(loss)

            #Create Saver
            saver = tf.train.Saver()

            #--Main Session--#
            with tf.Session() as sess:
                saver.restore(sess, "Recurrent_Network/Saves/my_honours_project_saver.ckpt")#Restore Files
                print("---Testing Model---")#Prompter
                for iteration in range(self.num_batch):
                    x_batch, y_batch = test_next_batch()#Testing Inject
                    _, output_val, mse = sess.run([training_op, outputs, loss], feed_dict={self.A: x_batch, self.B: y_batch})
                    print(output_val)

    def test_next_batch(self):
        '''
        Get Next batch for placeholders inside network for Testing
        '''
        #Variables
        data = []
        data = np.zeros(shape=(self.n_steps,self.n_inputs))
        t_min = 0
        t_max = t_min + self.n_steps
        idx = np.arange(t_min,t_max)
        data = [data[i] for i in idx]
        x_batch = array(data).reshape(1,self.n_steps,self.n_inputs)
        #Return Batching
        return x_batch


        def train_next_batch(self):
        '''
        Get Next batch for placeholders inside network for Training
        '''
        #Variable Changers
        dataset_x = self.label_data
        dataset_y = self.label_data

        #Variables
        x_batch = []
        y_batch = []
        issue_warn = False

        #The lord have mercy upon this horrific code:
        #Checker
        temp_checker = self.current_pos + self.n_steps
        if temp_checker > self.num_data:
            #Issue
            issue_warn = True
        else:
            #No Issue
            issue_warn = False

        #Main loop
        #x_batch
        if self.current_pos == 0:#If Started
            t_min = 0#Set Min
            t_max = t_min + self.n_steps#Set Max
            idx = np.arange(t_min,t_max)
            x_batch = [dataset_x[i] for i in idx]
            y_batch = [dataset_y[i] for i in idx]
            #Position Index in statements
            self.current_pos = self.current_pos * 0#Reset
            self.current_pos = self.current_pos + t_max#Set for next

        elif issue_warn == False and self.current_pos != 0:#Regular Batch
            t_min = self.current_pos#Set Min
            t_max = t_min + self.n_steps#Set Max
            idx = np.arange(t_min,t_max)
            x_batch = [dataset_x[i] for i in idx]
            y_batch = [dataset_y[i] for i in idx]

            #Position Index in statements
            self.current_pos = self.current_pos * 0#Reset
            self.current_pos = self.current_pos + t_max#Set for next
            self.t_backup = self.t_backup * 0#Reset
            self.t_backup = self.current_pos#Setnew

        elif issue_warn == True and self.current_pos != 0:#If T_Max over limit
            #t_min = self.t_backup
            t_min = self.num_data - self.n_steps
            t_max = self.num_data#Set Maximum
            idx = np.arange(t_min,t_max)
            x_batch = [dataset_x[i] for i in idx]
            y_batch = [dataset_y[i] for i in idx]

        #Reshape
        x_batch = array(x_batch).reshape(1,self.n_steps,self.n_inputs)
        y_batch = array(y_batch).reshape(self.n_steps,self.n_outputs)
        #Return Batching
        return x_batch, y_batch