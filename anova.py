#!/usr/bin/env python
# coding: utf-8


from scipy import stats
from IPython.display import display
import numpy as np
import pandas as pd



def ANOVA1_partition_TSS(data):
    '''
    This function partitions the sum of squares in a one way ANOVA layout.
    The function takes a data set Xi,j for j = 1, . . . ,ni and i = 1, . . . , I, and return
    SStotal, SSw, and SSb

    SStotal = SSw + SSb.

    '''
    #calculate sum of squares within treatments
    SSw = 0
    for i in range(len(data)):
        meanTreatment = np.mean(data[i]) #sample mean of a treatment
        for j in range(len(data[i])):
            SSw += (data[i][j] - meanTreatment)**2
    
    #calculate sum of squares between treatments
    SSb = 0
    meanSample = np.mean([np.mean(treat) for treat in data])
    for i in range(len(data)):
        meanTreatment = np.mean(data[i]) #sample mean of a treatment
        SSb += (meanTreatment - meanSample)**2
    
    SStotal = SSw + SSb
    
    return {"SStotal":SStotal,"SSw": SSw,"SSb": SSb}



def ANOVA1_test_equality(data,signifLevel = 0.05):
    '''
    This function tests the equality of the means in a one way ANOVA layout.
    The input is a data set Xi,j for j = 1, . . . , ni and i = 1, . . . , I, and the significance
    level α. As the output, the function prints all the quantities in the table as well as the critical value, 
    the p-value, and the decision.
    '''
    n = sum([len(x) for x in data]) #total number of points
    
    dofWithin = n-len(data) #degrees of fredom within groups
    dofBetween = len(data)-1 #degrees of fredom between groups
    
    SStotal, SSw, SSb = ANOVA1_partition_TSS(data).values()
    
    msBetween = SSb / dofBetween
    msWithin = SSw / dofWithin
    
    f_stats = msBetween / msWithin
    crit = stats.f.ppf(q=1-signifLevel, dfn=dofBetween, dfd=dofWithin)
    p_value = stats.f.cdf(f_stats, dfn=dofBetween, dfd=dofWithin)
    
    df_source = pd.DataFrame({"df":[dofBetween,dofWithin,dofBetween+dofWithin],"SS":[SSb,SSw,SStotal],
                              "MS":[msBetween,msWithin,None],"F":[f_stats,None,None]},index=["SSb","SSw","SStotal"])
    display(df_source)
    
    print("\nCritical value: {}".format(crit))
    print("p-value: {}".format(p_value))
    
    decision = "Reject the null hypothesis"
    if p_value<signifLevel:
        decision = "DONT Reject the null hypothesis"
    
    print("Decision: {}, H0, at the {} level of significance.".format(decision,signifLevel))


def ANOVA1_is_contrast(c):
    '''
    A linear combination c = (c1, . . . , cI ) of the means 
    is called a contrast if sum(c1, . . . , cI) =0.
    '''
    if sum(c) == 0:
        return True
    else: return False
    
def ANOVA1_is_orthogonal(group_sizes, c1, c2)
    '''
    Two contrasts constructed by c1, c2 ∈ C are said to be
    orthogonal if their dot product divided by group size at each point is zero.
    '''
    if not ANOVA1_is_contrast(c1) or not ANOVA1_is_contrast(c2):
        return "WARNING: One of the coefficient vectors is not a constrast!" 
    
    elif sum([c1[i]*c2[i]/n for i,n in enumerate(group_sizes)]) == 0:
        return True
    
   
    
    
    
    