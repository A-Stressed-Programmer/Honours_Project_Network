        #Main Session
        if training:
            '''
            Train Model with Dropout Wrapper
            '''
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
            with tf.Session() as sess:
                saver.restore(sess, "Recurrent_Network/Saves/my_honours_project_saver.ckpt")#Restore Files
                print("---Testing Model---")#Prompter
                for iteration in range(self.num_batch):
                    x_batch, y_batch = test_next_batch()#Testing Inject
                    _, output_val, mse = sess.run([training_op, outputs, loss], feed_dict={self.X: x_batch, self.Y: y_batch})
                    print(output_val)



                    #Main THREE#
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
            cells = [tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=1.0) for cell in layers]

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