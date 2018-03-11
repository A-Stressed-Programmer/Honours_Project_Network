#Ryan G Cobb
#Honours Project
import csv
import operator
#Import Plotter for graph generation
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

class Data_Parser():
    def __init__(self):
        '''
        Variable holders for definitions
        '''
        self.temp_target = "Recurrent_Network/Temp/target_temp.csv"

    def clear_csv(filename):
        '''
        Clear target [Filename.CSV] file of data
        '''
        while True:
            try:#Try/Catch
                print("Clearing Target: [", filename, "] of data!")#Prompter
                f = open(filename, "w+")#Open File
                f.close()#Close File
                print("Target: [", filename, "] Cleared!")
            except Exception as e:#Exception Block
                print("Clear_CSV() Error: {0}]".format(str(e.args[0])).encode("utf-8"))

    def get_targets(self, data):
        '''
        Get Targets from current array by dropping first value of Input Array
        '''
        clear_csv(self.temp_target)#clear csv
        while True:
            try:#Try/Catch Errors
              print("Running: Get_Targets()!")#Prompter
              targets = []#Declare Target Array
              with open(self.temp_target, 'w') as csv_output:
                writer = csv.writer(csv_output, lineterminator='\n')#Writer
                data.pop(1)#Drop First Row
                targets = data#Append to Targets
                for val in targets:
                  writer.writerow(val)
                print("Get_Targets() Was Successful!")
                return targets#Return Targets to main

            except Exception as e:#Catch Errors
                print("Get_Targets() Error: {0}]".format(str(e.args[0])).encode("utf-8"))
                return None

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

    def normalize_data(self, input_data, label_data, n_inputs):
        '''
        Normalize dataset for injection into neural network utilizing the following formula
        
        normalize_data(self, input_data, label_data, n_inputs)

        Normalized_Value = (x - Min_Value)/(Max_Value - Min_Value)

        deNormalize;
        x = Xnorm x (Max_Value - Min_Value) + Min_Value
        '''
        #Vars
        #--Input Data--#
        temp=[]
        temp=np.ravel(input_data)#Ravel 2-D Array to 1-D array
        input_max_value = np.float(max(temp))
        input_min_value = np.float(min(temp))
        #--Label Data--#
        list=np.ravel(label_data)
        label_max_value = np.float(max(list))
        label_min_value = np.float(min(list))

        normalized_input = []#Storage for Normalized data
        normalized_label = []#Storage for label

        #Main loop
        for i in range(len(temp)):
            '''
            Normalized_Value = (x - Min_Value)/(Max_Value - Min_Value)
            '''
            #--Current Convert--#
            current = np.float(temp[i])
            #--Input Data--#
            sum_input = ((current-input_min_value)/(input_max_value-input_min_value))#Summation
            normalized_input.append(sum_input)#Append

        for i in range(len(label_data)):
            #--Label (1D Array)--#
            positon = np.float(list[i])
            sum_label = ((positon-label_min_value)/(label_max_value-label_min_value))
            normalized_label.append(sum_label)#Append Answer to new Array

        #IF Statement
        if n_inputs == 3:
            normalized_input = np.reshape(normalized_input,(-1,3))#Reshape to correct size
            #Return Normalized Arrays
            #RETURN MAX MIN as they will not persist in this 
            return normalized_input, normalized_label#Return dataset
        if n_inputs == 1:
            #RETURN MAX MIN as they will not persist in this 
            return normalized_input, normalized_label#Return dataset
        else:
            ValueError("You've got a problem Scotty!")

    def denomralize_data(self, norm_data):
        '''
        Denomalize dataset for results printing from neural network
        Normalized_Value = (x - Min_Value)/(Max_Value - Min_Value)

        deNormalize;
        x = Xnorm x (Max_Value - Min_Value) + Min_Value
        '''
        #Vars
        #--Input Data--#
        temp=[]
        temp=np.ravel(input_data)#Ravel 2-D Array to 1-D array
        input_max_value = np.float(max(temp))
        input_min_value = np.float(min(temp))
        #--Label Data--#
        list=np.ravel(label_data)
        label_max_value = np.float(max(list))
        label_min_value = np.float(min(list))

        normalized_input = []#Storage for Normalized data
        normalized_label = []#Storage for label

        #Main loop
        for i in range(len(temp)):
            '''
            Normalized_Value = (x - Min_Value)/(Max_Value - Min_Value)
            '''
            #--Current Convert--#
            current = np.float(temp[i])
            #--Input Data--#
            #sum_input = (current(i) - )#Summation
            normalized_input.append(sum_input)#Append

        for i in range(len(label_data)):
            #--Label (1D Array)--#
            positon = np.float(list[i])
            sum_label = ((positon-label_min_value)/(label_max_value-label_min_value))
            normalized_label.append(sum_label)#Append Answer to new Array

        normalized_input = np.reshape(normalized_input,(-1,3))#Reshape to correct size
        return normalized_input, normalized_label#Return dataset