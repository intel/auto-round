# collect diffusion input for calibration
global diffusion_input_list
diffusion_input_list = []

def appendInput(value):
    diffusion_input_list.append(value)

def getInputList():
    return diffusion_input_list


## collect quantization error for correction
global data_error_t_list
data_error_t_list = []  ## storing model out['mean'] and quantization error and its step t in format (data, error, t)
def append(value):
    data_error_t_list.append(value)

def getList():
    return data_error_t_list