
import datetime 
import os 

def set_experiment_dir(name:str):
    experiment_id = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
    directory = './'+experiment_id+'_'+name

    try : 
        if not os.path.exists(directory) :
            os.makedirs(directory)

    except OSError : 
            print ('Error: Creating directory. ' +  directory)
    
    return directory