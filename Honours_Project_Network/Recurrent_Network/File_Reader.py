import csv
import random
import numpy as np

class File_Reader():
    def __init__(self):
        '''
        Variable holders for folders
        '''
        #Folder grabbers for Temp 
        self.temp_label = "Recurrent_Network/Temp/full_label_data.csv"
        self.temp_train = "Recurrent_Network/Temp/full_input_data.csv"
        self.temp_x_axis = "Recurrent_Network/Temp/x_axis_data.csv"

    def clear_csv(filename):
        '''
        Clear target [Filename.CSV] file of data
        '''
        while True:
            try:#Try/Catch
                print("Clearing Target: [", filename, "] of data!")#Prompter
                f = open(filename, "w+")#Open File
                f.close()#Close File
                print("Target: [", filename, "] Cleared SUCCESSFULLY!")
                return None
            except Exception as e:#Exception Block
                print("Clear_CSV() ERROR: {0}]".format(str(e.args[0])).encode("utf-8"))
                return None

    def get_targets(self, data):
        '''
        Get Targets from current array by dropping first value of Input Array
        '''
        while True:
            try:#Try/Catch Errors
              print("RUNNING: Get_Targets()!")#Prompter
              File_Reader.clear_csv(self.temp_label)#clear csv
              targets = []#Declare Target Array
              with open(self.temp_label, 'w') as csv_output:
                writer = csv.writer(csv_output, lineterminator='\n')#Writer
                data.pop(0)#Drop First Row
                targets = data#Append to Targets
                for val in targets:
                  writer.writerow(val)
                print("Get_Targets() Was SUCCESSFUL!")
                return targets#Return Targets to main

            except Exception as e:#Catch Errors
                print("Get_Targets() ERROR: {0}]".format(str(e.args[0])).encode("utf-8"))
                return None

    def read_csv(self, filename):
        '''
        Read Csv file into number array, reads row by row without exception.
        [Column Zero Only!]
        '''
        #Error Handling
        while True:
            try:#Try Block+
                print("Attempting read_csv(" +filename+ ")")#Feedback
                full_path = ("Datasets/" + filename)#Path Fixer
                with open(full_path, 'r') as csv_input:#Open Csv
                    with open(self.temp_train, 'w') as csv_output:
                        writer = csv.writer(csv_output, lineterminator='\n')#Writer
                        reader = csv.reader(csv_input)#Reader
                        #Variables
                        attrs = []#Declare Array
                        rows = []#Secondary Array
                        #Main Loop
                        for row in reader:#For each row Do:
                          rows.append(row[0])#Append attrs row
                          attrs.append(row)
                        #Write to temp log
                        writer.writerows(attrs)#Log in temp
                #Print Success
                print("SUCCESSFULLY parsed data into Attrs Array!")
                return attrs#Return Array
            except Exception as e:#Exception Block
                print("Read_CSV() ERROR: {0}]".format(str(e.args[0])).encode("utf-8"))
                return None

    def row_writer(self, x_axis_data, input_data, label_data):
        '''
        Writes rows to temp files into Temp/Data Files.
        '''
        #Clear
        File_Reader.clear_csv(self.temp_train)
        File_Reader.clear_csv(self.temp_label)
        File_Reader.clear_csv(self.temp_x_axis)

        #Append X axis data for plotter to file
        with open(self.temp_x_axis, 'w') as csv_output:
            writer = csv.writer(csv_output, lineterminator='\n')#Writer
            writer.writerows(x_axis_data)

        #Append Input data to temp file
        with open(self.temp_train, 'w') as csv_output:
            writer = csv.writer(csv_output, lineterminator='\n')#Writer
            writer.writerows(input_data)

        #Append label_data to temp file
        with open(self.temp_label, 'w') as csv_output:
            writer = csv.writer(csv_output, lineterminator='\n')#Writer
            writer.writerows(label_data)

    def parse_data(self, filename):
        '''
        Process data into segments for datasets from the new website datasets.
        [Data, Open, High, Low, Close, Adj_close, Volume]
        Warning: No Successfully implemented ajustment for Null values, Remove Manually!
        '''
        full_path = ("Datasets/"+filename)
        while True:
            try:#Try/Catch sets
                with open(full_path, 'r') as csv_input:#Intialize
                    reader = csv.reader(csv_input)#Reader
                    next(csv_input) # skip header line
                    #Arrays
                    plotter_x_axis_data = []
                    input_data = []
                    label_data = []
                    #Define Grabbable sets
                    one_included_cols = [0]
                    two_included_cols = [1, 2, 3]
                    three_included_cols = [4]

                    #Main Loop
                    for row in reader:
                        data_1 = list(row[i] for i in one_included_cols)#Grab plotter data
                        data_2 = list(row[i] for i in two_included_cols)#Grab Input data
                        data_3 = list(row[i] for i in three_included_cols)#Grab Close data

                        #Append data
                        plotter_x_axis_data.append(data_1)#Plotter x_axis
                        input_data.append(data_2)#Input data for network
                        label_data.append(data_3)#Label data for accuarcy data

                #Store data to temp files
                File_Reader.row_writer(self, plotter_x_axis_data, input_data, label_data)
                #Return datasets
                return plotter_x_axis_data, input_data, label_data

            #Exception
            except Exception as e:
                print("Parse_Data() ERROR: {0}]".format(str(e.args[0])).encode("utf-8"))#Return Data
                break

    def read_into_temp(self, data, filename):
        '''
        Read Results data(x) into file (target_temp) for storage
        Requires format, i.e. "/RNN/error_results.csv"
        '''
        #Append Full file path
        full_path = ("Recurrent_Network/Temp"+filename)
        #Clear Current CSV
        File_Reader.clear_csv(full_path)
        #Append X axis data for plotter to file
        with open(full_path, 'w') as csv_output:
            writer = csv.writer(csv_output, lineterminator='\n')#Writer
            writer.writerows(map(lambda x: [x], data))




        




