
import json

from tensorflow.python.ops.metrics_impl import false_negatives

from echantillonneur import echantillonner
from formater import formate_datas
from filter import filter_datas , filter_datas_to_player
#
# with open('params.json', "r", encoding="utf-8") as f:
#     data = json.load(f)
# params = data['params']
# result =   data['result']
#
# ################################################################################
# path_train  = data['path_formater_to_train']['data_to_train']
# path_filtered_data_train = data['path_formater_to_train']['data_formated']
#
# path_input_play = data['path_data_play']['data_input']
# path_save_play = data['path_data_play']['data_output']
# data_to_play = data['path_formater_to_play']['data_to_play']
# path_of_formated_data_play = data['path_formater_to_play']['data_formated']
#
#
# data_input = data['path_data_train']['data_input']
# data_output = data['path_data_train']['data_output']
# data_brain = data['path_brain']
# path_params_brains = data['path_param_brains']
# name_brain = data['name_brain']
# epochs_brain = data['epoch_brain']
#
# formated_data = data['path_formater_to_train']['data_formated']
# path_entry = data['path_data_train']['data_input']
# path_out = data['path_data_train']['data_output']
#
#
#
# path_inputs_data = f'{data_input}' #pour entrainement
# #print(path_inputs_data)
# path_outputs_data = f'{data_output}' #pour entrainement
# path_brain = f"{data_brain}"
# input_shape = data['input_shape']
# output_shape = data['output_shape']
#
#
# ##################################################################################
# ec = data['echantilloner']
# time_echantion =data['time_echantion']
#
# e_train = data['e_data']['e_train']
# f_train = data['e_data']['f_train']
#
#
# e_test = data['e_data']['e_test']
# f_test = data['e_data']['f_test']
#
# e_play = data['e_data']['e_play']
# f_play = data['e_data']['f_play']
#
# time_echantion = data['time_echantion']

# def smart_train (path_train):
#     if ec == True:
#         try :
#             print("smart frame activated to training")
#             echantillonner(e_train , time_echantion , f_train)
#             #echantillonner(e_test, time_echantion, f_test)
#             #echantillonner(e_play, time_echantion, f_play)
#         except Exception as e:
#             print(e)
#     formate_datas(path_train , path_filtered_data_train)    #train
#     filter_datas(formated_data , path_entry ,path_out)      #train
#
# def smart_test (path_test):
#     if ec == True:
#         try :
#             print("smart frame activated to testing")
#             #echantillonner(e_train , time_echantion , f_train)
#             echantillonner(e_test, time_echantion, f_test)
#             #echantillonner(e_play, time_echantion, f_play)
#         except Exception as e:
#             print(e)
#     formate_datas(path_train , path_filtered_data_train)    #train
#     filter_datas(formated_data , path_entry ,path_out)      #train
#
#
# def smartf_play ():
#     if ec == True:
#         try :
#             print("smart frame activated to player")
#             #echantillonner(e_train , time_echantion , f_train)
#             #echantillonner(e_test, time_echantion, f_test)
#             echantillonner(e_play, time_echantion, f_play)
#         except Exception as e:
#             print(e)
#     formate_datas(data_to_play, path_of_formated_data_play)
#     filter_datas_to_player(path_of_formated_data_play, path_input_play)


class SmartFrame :
    def __init__(self , ech, time_echantion) :
        self.echantillonner = ech
        self.time_echantion = time_echantion

    def training_data (self,e_train , path_train ,f_train , path_filtered_data_train ,formated_data ,path_entry , path_out):
        if self.echantillonner == True:
            try :
                print("smart frame activated to training")
                echantillonner(e_train , self.time_echantion , f_train)
                #echantillonner(e_test, time_echantion, f_test)
                #echantillonner(e_play, time_echantion, f_play)
            except Exception as e:
                print(e)


        formate_datas(path_train , path_filtered_data_train)    #train
        filter_datas(formated_data , path_entry ,path_out)      #train

    def testing_data(self,e_test,path_test , f_test  , path_filtered_data_test , formated_data ,path_entry , path_out ) :
        if self.echantillonner == True:
            try:
                print("smart frame activated to testing")
                # echantillonner(e_train , time_echantion , f_train)
                echantillonner(e_test, self.time_echantion, f_test)
                # echantillonner(e_play, time_echantion, f_play)
            except Exception as e:
                print(e)
        formate_datas(path_test, path_filtered_data_test)  # train
        filter_datas(formated_data, path_entry, path_out)  # train

    def playing_data(self , e_play, time_echantion , f_play , data_to_play , path_of_formated_data_play , path_input_play):
        if (echantillonner== True):
            try:
                print("smart frame activated to player")
                # echantillonner(e_train , time_echantion , f_train)
                # echantillonner(e_test, time_echantion, f_test)
                echantillonner(e_play, time_echantion, f_play)
            except Exception as e:
                print(e)
        formate_datas(data_to_play, path_of_formated_data_play)
        filter_datas_to_player(path_of_formated_data_play, path_input_play)

