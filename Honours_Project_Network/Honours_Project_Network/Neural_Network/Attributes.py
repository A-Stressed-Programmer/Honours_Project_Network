'''
--Attributes--
Holds main attributes for the LSTM Deep Network
Values include;
-Layer nodes
-classes
-batch sizes
'''
class lstm_atr():
    '''
    Container class for the main attributes involved in the LSTM network; thusly including; layer nodes, batch size & classes.
    '''
    rnn_size = 128 #universal size of node count in layer

    #Nodes
    layer_one_nodes = rnn_size #Hidden Layer one hidden Nodes
    layer_two_nodes = rnn_size #Hidden Layer two hidden Nodes
    layer_three_nodes = rnn_size #Hidden Layer three hidden Nodes

    number_classes = 10
    batch_size = 100 #Batch size for Batch Processing for large datasets
