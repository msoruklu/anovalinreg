from scipy import stats
from IPython.display import display
import numpy as np
import pandas as pd
from anova import *
from linear_regression import *
import data_for_ANOVA as ANOVA
import data_for_LR as LR

data = ANOVA.data_anova1
alpha = 0.1 #set significance level 
c1 =[1,0,0,-1]
c2 = [0,-1,1,0]
C = [c1,c2]

print("Printing the ANOVA function's results with the given data.\n\n")
print("data: " ,str(data))

print(ANOVA1_partition_TSS(data))
print("--------------------------------------------------------------\n")
ANOVA1_test_equality(data,alpha)
print("--------------------------------------------------------------\n")
print("c1 is a contrast: ",ANOVA1_is_contrast(c1))
print("c2 is a contrast: ",ANOVA1_is_contrast(c2))
print("--------------------------------------------------------------\n")
group_sizes =[len(i) for i in data] #size of each treatment as a list
print("c1: ",c1)
print("c2: ",c2)
print("c1 and c2 are orthogonal: ", ANOVA1_is_orthogonal(group_sizes,c1,c2))
print("--------------------------------------------------------------\n")
print("Bonferroni:",Bonferroni_correction(alpha, len(C)))
print("--------------------------------------------------------------")
print("Sidak:",Sidak_correction(alpha, len(C)))
print("--------------------------------------------------------------")
print("Scheffe:",ANOVA1_CI_linear_combs(data,C,method="Scheffe",signifLevel=alpha))  
print("--------------------------------------------------------------")
print("Bonferroni:",ANOVA1_CI_linear_combs(data,C,method="Bonferroni",signifLevel=alpha)) 
print("--------------------------------------------------------------")   
print("Sidak:",ANOVA1_CI_linear_combs(data,C,method="Sidak",signifLevel=alpha))    
print("--------------------------------------------------------------")   
print("Tukey:",ANOVA1_CI_linear_combs(data,C,method="Tukey",signifLevel=alpha))  
print("--------------------------------------------------------------")     
print("best:",ANOVA1_CI_linear_combs(data,C,method="best",signifLevel=alpha))    

print("--------------------------------------------------------------\n")
print("Scheffe:",ANOVA1_test_linear_combs(data,C,[0]*len(C),FWER=alpha,method="Scheffe"))
print("--------------------------------------------------------------") 
print("Bonferroni:",ANOVA1_test_linear_combs(data,C,[0]*len(C),FWER=alpha,method="Bonferroni"))
print("--------------------------------------------------------------") 
print("Sidak:",ANOVA1_test_linear_combs(data,C,[0]*len(C),FWER=alpha,method="Sidak"))
print("--------------------------------------------------------------") 
print("Tukey:",ANOVA1_test_linear_combs(data,C,[0]*len(C),FWER=alpha,method="Tukey"))
print("--------------------------------------------------------------") 
print("best:",ANOVA1_test_linear_combs(data,C,[0]*len(C),FWER=alpha,method="best"))
print("End of ANOVA 1")
print("--------------------------------------------------------------\n\n")

data = ANOVA.data_anova2
print("Printing the result for function: {}".format("ANOVA2_partition_TSS"))
print(ANOVA2_partition_TSS(data))
print("--------------------------------------------------------------\n")
print("Printing the result for function: {}".format("ANOVA2_MLE"))
print(ANOVA2_MLE(data))
print("--------------------------------------------------------------\n")
print("choice: A")
ANOVA2_test_equality(data, alpha, choice = "A")
print("choice: B")
ANOVA2_test_equality(data, alpha, choice = "B")
print("choice: AB")
ANOVA2_test_equality(data, alpha , choice = "AB")
print("--------------------------------------------------------------\n")
print("End of ANOVA 2\n\n\n")
#import the red winequality dataset --> quality is the response variable

data = pd.read_csv('test_data/winequality-red.csv',delimiter=",")
features = list(data.columns)
features.insert(0,"beta_zero")
print("Features in the dataset:\n {}\n".format(features))
data=data.values

alpha=0.05
design = data[:,:-1]
#inset  column of ones
design = np.insert(design, 0, np.ones(len(design)), axis=1)
response = data[:,-1] #quality at the last column
C = np.identity(len(design[0]-1))
c_zero = np.ones(len(design[0]-1))

print("Printing the LINEAR REGRESSION function's results with the given data.\n\n")
print("design: " ,str(design))
print("response: ",response)
print("--------------------------------------------------------------\n")
print("Printing the result for function: {}".format("Mult_LR_Least_squares"))
print(Mult_LR_Least_squares(design,response))
print("--------------------------------------------------------------\n")
print("Printing the result for function: {}".format("Mult_LR_partition_TSS"))
pprint(Mult_LR_partition_TSS(design,response))
print("--------------------------------------------------------------\n")
print("Printing the result for function: {}".format("Mult_norm_LR_simul_CI"))
pprint(Mult_norm_LR_simul_CI(design,response,alpha))
print("--------------------------------------------------------------\n")
print("Printing the result for function: {}".format("Mult_norm_LR_CR"))
pprint(Mult_norm_LR_CR(design,response,C,alpha))
print("--------------------------------------------------------------\n")
print("Printing the result for function: {}".format("Mult_norm_LR_is_in_CR"))
pprint(Mult_norm_LR_is_in_CR(design,response,C,c_zero,alpha))
print("--------------------------------------------------------------\n")
print("Printing the result for function: {}\n".format("Mult_norm_LR_test_general"))
print("\nTesting the hypothesis that {} has no impact on the quality.".format(features[6]))
print("H0: β6=0, H1: not H0 \n")
C = [[0,0,0,0,0,0,1,0,0,0,0,0]]
c_zero = [0]
Mult_norm_LR_test_general(design,response,C,c_zero,alpha)
print("--------------------------------------------------------------\n")
print("Printing the result for function: {}".format("Mult_norm_LR_test_comp"))
print("\nTesting the hypothesis that {}, {}, {} and {} has no impact on the quality.".format(features[1],features[3],features[4],features[9]))
print("H0: β1=β3=β4=β9=0, H1: not H0 \n")
J = [1,3,4,9]
Mult_norm_LR_test_comp(design,response,J,alpha)
print("--------------------------------------------------------------\n")
print("Printing the result for function: {}".format("Mult_norm_LR_test_linear_reg"))
print("\nTesting the hypothesis that there exists a linear regression at all.")
print("H0: All β's are 0. H1: not H0")
Mult_norm_LR_test_linear_reg(design,response,alpha)
print("--------------------------------------------------------------\n")
D= [[1,2,3,4,456,5,6,6,7,8,250,16],
    [1,78,6,14,5,15,6,40,7,100,26,16]]
print("Printing the result for function: {} for different methods.".format("Mult_norm_LR_pred_CI"))
print( "Scheffe: \n")
pprint(Mult_norm_LR_pred_CI(design,response,D,alpha,"Scheffe"))
print("--------------------------------------------------------------\n") 
print( "Bonferroni: \n")
pprint(Mult_norm_LR_pred_CI(design,response,D,alpha,"Bonferroni"))
print("--------------------------------------------------------------") 
print("Best: \n")
pprint(Mult_norm_LR_pred_CI(design,response,D,alpha,"best"))
print("--------------------------------------------------------------\n")

print("End of all functions. You can try with your own data as well.")