
### -------------------------# Imports #------------------------------- ###

from echantillonneur import echantillonner
from formater import formate_datas
from filter import filter_datas , filter_datas_to_player

### ------------------------------------------------------------------- ###




class SmartFrame :
    def __init__(self , ech, time_echantion , delimiter) :
        self.echantillonner = ech
        self.time_echantion = time_echantion
        self.delimiter = delimiter

    def training_data (self,e_train , path_train ,f_train , path_filtered_data_train ,formated_data ,path_entry , path_out):
        if self.echantillonner == True:
            try :
                print(f"smart frame activated to training : ech {self.echantillonner} time {self.time_echantion} ")
                echantillonner(e_train , self.time_echantion , f_train , self.delimiter)
            except Exception as e:
                print(e)


        formate_datas(path_train , path_filtered_data_train , self.delimiter)    #train
        filter_datas(formated_data , path_entry ,path_out , self.delimiter)      #train

    def testing_data(self,e_test,path_test , f_test  , path_filtered_data_test , formated_data ,path_entry , path_out ) :
        if self.echantillonner == True:
            try:
                print("smart frame activated to testing")
                echantillonner(e_test, self.time_echantion, f_test , self.delimiter)
            except Exception as e:
                print(e)
        formate_datas(path_test, path_filtered_data_test , self.delimiter)  # train
        filter_datas(formated_data, path_entry, path_out , self.delimiter)  # train

    def playing_data(self , e_play, f_play , data_to_play , path_of_formated_data_play , path_input_play):
        if self.echantillonner== True :
            try:
                print("smart frame activated to player")

                echantillonner(e_play, self.time_echantion, f_play)
            except Exception as e:
                print(e)
        formate_datas(data_to_play, path_of_formated_data_play)
        filter_datas_to_player(path_of_formated_data_play, path_input_play)

