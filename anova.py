#!/usr/bin/env python
# coding: utf-8

# In[18]:


from scipy import stats
from IPython.display import display
import numpy as np
import pandas as pd


# In[19]:


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


# In[66]:


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


# In[75]:


data=[[28,23,14,27,31,24],
      [33,36,34,29,24],
      [18,21,20,22],
      [11,14,11,16]]
SStotal, SSw, SSb = ANOVA1_partition_TSS(data).values()


# In[76]:


ANOVA1_test_equality(data)


# In[65]:



crit = 2.8450678052793514
stats.f.cdf(crit, dfn=3, dfd=39)


# In[39]:


stats.f.ppf(q=1-0.05, dfn=3, dfd=39)


# In[ ]:




