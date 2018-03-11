#Project Imports
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

class Data_Plotter():
    '''
    Plotter houses main MatPlotLib.Extenstion(PythonPlot) commands to visualize datasets

    Initialize main variables for Plotter;
    RNN(self, n_neurons, n_layers, learning_rate, fn_select, filename, epoch)
    '''

    def model_selector(nn_type):
        '''
        Take user selector return file user data
        '''
        #SPLIT RNN/LSTM
        if nn_type == 'rnn':
            '''
            '''
            information = "RNN"
            save_file = "Recurrent_Network/Graphs/" + information + "/Single/train_data.png"
        if nn_type == 'lstm':
            '''
            '''
            information = "LSTM"
            save_file = "Recurrent_Network/Graphs/" + information + "/Single/train_data.png"

        if nn_type == 'bi-rnn':
            '''
            '''
            information = "BI-RNN"
            save_file = "Recurrent_Network/Graphs/" + information + "/Single/train_data.png"

        if nn_type == 'bi-lstm':
            '''
            '''
            information = "BI-LSTM"
            save_file = "Recurrent_Network/Graphs/" + information + "/Single/train_data.png"

        #Return details for plotters
        return information, save_file

#--------------------------------------------------------------------------#

    def input_graph(self,input_data, filename):
        '''
        Plot Input data graph from inputted datasets and append to local storage;

        input_graph(self, input_data, filename):
        '''
        while True:
            try:#Try/Catch for error Running
                print("Running [Input_Graph()]!")
                title = ("Initial Input Data:[" + filename + "]")
                plt.xlabel("Trading Days")
                plt.ylabel("Single Stock Value(£)")
                plt.title(title,color="red", fontsize=20)#Plot Title

                plt.plot(input_data,linestyle='-',linewidth=2,color="red" ,label="Label Data")#Plot Data
                plt.legend(loc="upper left")#Lengend
                plt.savefig("Recurrent_Network/Graphs/input_data.png")#Save me
                plt.close()
                success = "Plot Completed Successfully!"
                print(success)
                return success#Return 
            except Exception as e:#Exception for Errors
                plt.close()
                exception = ("Input_Graph() ERROR: {0}]".format(str(e.args[0])).encode("utf-8"))
                print(exception)
                return exception#Return 

#--------------------------------------------------------------------------#

    def train_graph(self,filename, output_data, nn_type):
        '''
        Plot Training data graph from network output data and append to local storage;

        train_graph(self,filename, output_data, nn_type):
        '''

        #Get details for plotters
        information, save_file = Data_Plotter.model_selector(nn_type)

        while True:
            try:#Try/Catch for error Running
                print("Running [Train_Graph()]!")
                title = ("Training("+ information +"):["+ filename + "]")
                plt.xlabel("Trading Days")
                plt.ylabel("Single Stock Value(£)")
                plt.title(title, color="blue", fontsize=14)
                plt.plot(output_data, linestyle='-',linewidth=2, color="blue", label="Predicted Data")#Plot Predicted
                plt.legend(loc="upper left")
                plt.savefig(save_file)#Save to File
                plt.close()
                success = "Plot Completed Successfully!"
                print(success)
                return success#Return 
            except Exception as e:#Exception for Errors
                plt.close()
                exception = ("Train_Graph() ERROR: {0}]".format(str(e.args[0])).encode("utf-8"))
                print(exception)
                return exception#Return DONE

    def test_graph(self,filename, output_data, nn_type):
        '''
        Plot Testing data graph from network output data and label data and append to local storage

        test_graph(self, output_data, label_data):
        '''

        #Get details for plotters
        information, save_file = Data_Plotter.model_selector(nn_type)

        while True:
            try:#Try/Catch for error Running
                print("Running [test_graph()]!")
                title = ("Testing("+information+"):["+ filename + "]")
                plt.xlabel("Trading Days")
                plt.ylabel("Single Stock Value(Normalized)")
                plt.title(title, color="green", fontsize=14)
                plt.plot(output_data, linestyle='-',linewidth=2, color="green", label="Predicted Data")#Plot Predicted
                plt.legend(loc="upper left")
                plt.savefig(save_file)
                plt.close()
                success = "Plot Completed Successfully!"
                print(success)
                return success#Return 
            except Exception as e:#Exception for Errors
                plt.close()
                exception = ("Test_Graph() ERROR: {0}]".format(str(e.args[0])).encode("utf-8"))
                print(exception)
                return exception#Return DONE

#--------------------------------------------------------------------------#

    def c_train_graph(self,filename, output_data, label_data, nn_type):
        '''
        Plot Training data graph from network output data and label data and append to local storage;

        train_graph(self,filename, output_data, label_data):
        '''

        #Get details for plotters
        information, save_file = Data_Plotter.model_selector(nn_type)

        while True:
            try:#Try/Catch for error Running
                print("Running [c_train_graph()]!")
                title = ("Training Compare("+information+"):["+ filename + "]")
                plt.xlabel("Trading Days")
                plt.ylabel("Single Stock Value(Normalized)")
                plt.title(title, color="blue", fontsize=14)
                plt.plot(output_data, linestyle='-',linewidth=1, color="blue", label="Training Prediction")#Plot Predicted
                plt.plot(label_data, linestyle='-',linewidth=1, color="red", label="Actual Data")#Plot Actual
                plt.legend(loc="upper left")
                plt.savefig(save_file)
                plt.close()
                success = "Plot Completed Successfully!"
                print(success)
                return success#Return 
            except Exception as e:#Exception for Errors
                plt.close()
                exception = ("Train_Graph() ERROR: {0}]".format(str(e.args[0])).encode("utf-8"))
                print(exception)
                return exception#Return

    def c_test_graph(self,filename, output_data, label_data, nn_type):
        '''
        Plot Testing data graph from network output data and label data and append to local storage;

        c_test_graph(self, output_data, label_data):
        '''

        #Get details for plotters
        information, save_file = Data_Plotter.model_selector(nn_type)

        while True:
            try:#Try/Catch for error Running
                plt.close()
                print("Running [c_test_graph()]!")
                title = ("Testing Compare("+information+"):["+ filename + "]")
                plt.xlabel("Trading Days")
                plt.ylabel("Single Stock Value(Normalized)")
                plt.title(title, color="blue", fontsize=14)
                plt.plot(output_data, linestyle='-',linewidth=1, color="blue", label="Training Prediction")#Plot Predicted
                plt.plot(label_data, linestyle='-',linewidth=1, color="red", label="Actual Data")#Plot Actual
                plt.legend(loc="upper left")
                plt.savefig(save_file)
                plt.close()
                success = "Plot Completed Successfully!"
                print(success)
                return success#Return 
            except Exception as e:#Exception for Errors
                plt.close()
                exception = ("Train_Graph() ERROR: {0}]".format(str(e.args[0])).encode("utf-8"))
                print(exception)
                return exception#Return

    def c_combined_graph(self,filename, train_label, test_label, train_data, test_data, nn_type):
        '''
        Compare Training and Testing Data;
        c_combined_graph(self,filename, train_label, test_label, train_data, test_data)
        '''

        #Get details for plotters
        information, save_file = Data_Plotter.model_selector(nn_type)

        while True:
            try:#Try/Catch for error Running
                print("Begining [c_combined_graph] Plotting")
                #Top Plot-Training
                plt.subplot(2,1,1)
                title = ("Training("+information+"):["+ filename + "]")
                plt.plot(train_label, linestyle='-',linewidth=1, color="g", label="Train Label", alpha=0.5)#Plot Acutal
                plt.plot(train_data, linestyle='-',linewidth=1, color="b", label="Training Prediction", alpha=0.5)#Plot Predicted
                plt.legend(loc="upper left")
                #Bottom Plot-Testing
                plt.subplot(2,1,2)
                title = ("Testing("+information+"):["+ filename + "]")
                plt.xlabel("Trading Days")
                plt.ylabel("Single Stock Value(Normalized)")
                plt.plot(test_label, linestyle='-',linewidth=2, color="g", label="Test Label", alpha=0.5)#Plot Acutal
                plt.plot(test_data, linestyle='-',linewidth=2, color="r", label="Testing Prediction", alpha=0.5)#Plot Predicted
                plt.legend(loc="upper left")
                #Commands
                plt.savefig(save_file)
                plt.close()
                print("Plot [c_combined_graph] Completed Successfully!")
                success = "Plot Completed Successfully!"
                return success#Return 
            except Exception as e:#Exception for Errors
                plt.close()
                exception = ("c_combined_graph ERROR: {0}]".format(str(e.args[0])).encode("utf-8"))
                print(exception)
                return exception#Return

#--------------------------------------------------------------------------#

    def plt_error_graph(self,filename, error_data, epochs, nn_type):
        '''
        Plot Measurement graph of Mean Square Error over Number of Total Epochs to runs sets and append to local storage

        error_graph(self,filename, error_data, epochs)
        '''

        #Get details for plotters
        information, save_file = Data_Plotter.model_selector(nn_type)

        #Fill array for prediction
        n_epochs=[]
        for i in range(epochs):
            n_epochs.append(i)

        while True:
            try:#Try/Catch for error Running
                print("Running [plt_error_graph()]!")
                title = ("Mean Squared Error("+information+"):["+ filename + "]")
                plt.xlabel("Number of Epochs")
                plt.ylabel("Error")
                plt.title(title, color="purple", fontsize=14)
                plt.plot(error_data, linestyle='-',linewidth=2, color="purple", label="MSE")#Plot Predicted
                #Global Cost Minimum
                #ymax = min(error_data)
                #xpos = error_data.index(ymax)
                #xmax = n_epochs[xpos]
                #full_data = ("Global Cost Minimum:[Value(" + ymax + "), Epoch Number(" + xmax + ")]")
                #plt.annotate(full_data, xy=(200, ymax), xytext=(200, ymax+5),arrowprops=dict(facecolor='black', shrink=0.05),)
                plt.legend(loc="upper left")
                plt.savefig(save_file)
                plt.close()
                success = "Plot Completed Successfully!"
                print(success)
                return success#Return 
            except Exception as e:#Exception for Errors
                plt.close()
                exception = ("plt_error_graph() ERROR: {0}]".format(str(e.args[0])).encode("utf-8"))
                print(exception)
                return exception#Return

#--------------------------------------------------------------------------#

    def plt_nn_train(self,label_data, rnn_data, lstm_data):
        '''
        Builds concatinated graph of all models current enrolled in Honours Project
        '''
        while True:
            try:#Try/Catch for error Running
                print("Running [plt_nn_train()]!")
                plt.xlabel("Trading Days")
                plt.ylabel("Single Stock Value(Normalized)")
                plt.title("Model Training Comparison", color="m", fontsize=15)
                plt.plot(rnn_data, linestyle='-',linewidth=2, color="b", label="Recurrent Neural Network")#Plot Predicted
                plt.plot(lstm_data, linestyle='-',linewidth=2, color="g", label="Long Short-Term Memory")#Plot Predicted
                plt.plot(label_data, linestyle='-',linewidth=2, color="r", label="Actual Data")#Plot Predicted
                plt.legend(loc="upper left")
                plt.savefig("Recurrent_Network/Graphs/nn_train.png")#Save to File
                plt.close()
                success = "Plot Completed Successfully!"
                print(success)
                return success#Return 
            except Exception as e:#Exception for Errors
                plt.close()
                exception = ("Train_Graph() ERROR: {0}]".format(str(e.args[0])).encode("utf-8"))
                print(exception)
                return exception#Return DONE

    def plt_nn_test(self,label_data, rnn_data, lstm_data):
        '''
        Builds concatinated graph of all models current enrolled in Honours Project
        '''
        while True:
            try:#Try/Catch for error Running
                print("Running [plt_nn_train()]!")
                plt.xlabel("Trading Days")
                plt.ylabel("Single Stock Value(Normalized)")
                plt.title("Model Testing Comparison", color="m", fontsize=15)
                plt.plot(rnn_data, linestyle='-',linewidth=2, color="b", label="Recurrent Neural Network")#Plot Predicted
                plt.plot(lstm_data, linestyle='-',linewidth=2, color="g", label="Long Short-Term Memory")#Plot Predicted
                plt.plot(label_data, linestyle='-',linewidth=2, color="r", label="Actual Data")#Plot Predicted
                plt.legend(loc="upper left")
                plt.savefig("Recurrent_Network/Graphs/nn_test.png")#Save to File
                plt.close()
                success = "Plot Completed Successfully!"
                print(success)
                return success#Return 
            except Exception as e:#Exception for Errors
                plt.close()
                exception = ("Train_Graph() ERROR: {0}]".format(str(e.args[0])).encode("utf-8"))
                print(exception)
                return exception#Return DONE
