        #Main session
        with tf.Session() as sess:
           if training:
               init.run()#Initialize
               #---Outer Epoch Shell---#
               print("---Training RNN Model---")
               for epoch  in range(self.epoch):
                   #---Inner Iteration shell----#
                   self.results = []*0
                   for iteration in range(self.num_batch):
                       #---Execution Shell---#
                       x_batch, y_batch = rnn.train_next_batch()
                       _, output_val, mse = sess.run([training_op, outputs, loss], feed_dict={self.X: x_batch, self.Y: y_batch})
                       #Append data to array
                       fucking_work = []
                       for i in range(len(output_val)):
                           fucking_work.append(output_val[i])
                       self.results.append(fucking_work)
                       self.loss.append(mse)
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
           else:
               '''
               Model Without dropout layer! 
               '''
               saver.restore(sess, "Recurrent_Network/Saves/my_honours_project_saver.ckpt")#Restore Files
               print("---Testing Model---")#Prompter
               #sequence = [[0.]* self.n_inputs] * self.n_steps
               for iteration in range(self.num_batch):
                   #---Inner Operation Shell---#
                   #x_batch = np.array(sequence[-self.n_steps:self.n_inputs]).reshape(1, self.n_steps, self.n_inputs)
                   x_batch, y_batch = test_next_batch()
                   y_pred = sess.run(outputs, feed_dict={self.X:x_batch, self.Y:y_batch})
                   print(y_pred)
                   #sequence.append(y_pred[0,-1,0])

                           #--Main Session--#
        with tf.Session() as sess:
            saver.restore(sess, "Recurrent_Network/Saves/my_honours_project_saver.ckpt")#Restore Files
            for epoch in range(1):
                self.test_results = []*0#Zero Array
                for iteration in range(self.num_batch):
                    #--Execution Shell--#
                    x_batch = rnn.test_next_batch()#Testing Inject
                    output_val = sess.run([outputs], feed_dict={X: x_batch})#Inject data into array
                    self.test_results.append(output_val#Append to array)
                    #--Execution Shell End--#
                #--Inner Shell End--#
                dataplotter = Data_Plotter()
                dataplotter.test_graph(self.results)


            saver.restore(sess, "Recurrent_Network/Saves/my_honours_project_saver.ckpt")#Restore Files
            print("---Testing Model---")#Prompter
            self.test_results = []*0#Zero Array
            for iteration in range(self.num_batch):
                #--Execution Shell--#
                x_batch = rnn.test_next_batch()#Testing Inject
                output_val = sess.run([outputs], feed_dict={X: x_batch})#Inject data into array
                self.test_results.append(output_val#Append to array)
                #--Execution End--#
            #--Inner Shell End--#




'''
            saver.restore(sess, "Recurrent_Network/Saves/my_honours_project_saver.ckpt")#Restore Files
            print("---Testing Model---")#Prompter
            self.test_results = []*0#Zero Array
            for iteration in range(self.num_batch):
                #--Execution Shell--#
                x_batch = rnn.test_next_batch()#Testing Inject
                output_val = sess.run([outputs], feed_dict={X: x_batch})#Inject data into array
                self.test_results.append(output_val#Append to array)
                #--Execution End--#
            #--Inner Shell End--#
            '''