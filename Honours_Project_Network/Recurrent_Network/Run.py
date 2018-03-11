from RNN_Model import RNN
from Data_Plotter import Data_Plotter
dataplotter = Data_Plotter()
#Imports main RNN class for Construction

#Variable Declare for both RNN, Bi_RNN, LSTM, Bi_LSTM
rrn = RNN(n_neurons=50,n_layers=5,learning_rate=1e-4,fn_select='tahn',filename='AAPL.csv',epoch=150,batches=1,percentage=0.75, toggle_num=2)#Initialize
'''
n_neurons = Number of RNN CELL NEURONS Per LAYER
n_layers = Number of Layers in Neural Network
learning_rate = Starting Learning Rate, RNN Utilizes Learning Decay from AdamOptimizer
fn_select = Activation Function Selection ()
filename = Filename from /Datasets/ Folder in format ("GOOG.csv")
epoch = Number of WHOLE Dataset Iterations
Batches = Number of Batches in Epoch, I.e. divided dataset within session. DO NOT OPERATE FOR RNN
percentage = Percentage split of Training/Testing Data, i.e. 0.95 = 95% of Dataset is TRAINING and 5% is Testing
toggle_num = Toggle Switch for injection type, Use [Open][High][Low]-->[Predict(Close)] on toggle 1 OR [Close]-->[Predict_Next(Close)] on toggle 2
'''
#--RNN--#
#rrn.train_rnn_model() 
#rrn.test_rnn_model()
#--BI-RNN--#

#--LSTM--#
#rrn.train_lstm_model()
#rrn.test_lstm_model()
#--BI-LSTM--#
rrn.train_bi_lstm_model()
#rrn.test_bi_lstm_model()




