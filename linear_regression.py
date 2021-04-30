import numpy as np

def Mult_LR_Least_squares(design,response):
    #design is X matrix and response is y
    beta = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.array(design).transpose(),np.array(design))),np.array(design).transpose()),response)
    predictions = np.matmul(np.array(design),beta)
    error = predictions-response
    sigmaSqr = sum(np.square(error))/(len(design)-len(design[0])-1)
    return {"beta":beta,"sigmaSqr":sigmaSqr}

