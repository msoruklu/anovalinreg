#!/usr/bin/env python
# coding: utf-8


from sys import intern
from scipy import stats
from scipy.stats.stats import median_absolute_deviation #to get t and f statistics 
from statsmodels.stats.libqsturng import psturng, qsturng #to get statistic from Studentized range distribution
from itertools import combinations #to get pairwise linear combinations
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
    SStotal = 0
    SSb = 0
    meanSample = np.mean([np.mean(treat) for treat in data])
    

    for i in range(len(data)):
        meanTreatment = np.mean(data[i]) #sample mean of a treatment
        for j in range(len(data[i])):
            SSw += (data[i][j] - meanTreatment)**2
            SStotal += (data[i][j] - meanSample)**2
    
    #calculate sum of squares between treatments 
    for i in range(len(data)):
        meanTreatment = np.mean(data[i]) #sample mean of a treatment
        SSb += len(data[i])*(meanTreatment - meanSample)**2
    
    
    
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
    p_value = 1-stats.f.cdf(f_stats, dfn=dofBetween, dfd=dofWithin)
    
    df_source = pd.DataFrame({"df":[dofBetween,dofWithin,dofBetween+dofWithin],"SS":[SSb,SSw,SStotal],
                              "MS":[msBetween,msWithin,None],"F":[f_stats,None,None]},index=["SSb","SSw","SStotal"])
    df_source = df_source.replace(np.nan, '', regex=True)
    print(df_source)
    
    print("\nCritical value: {}".format(crit))
    print("p-value: {}".format(p_value))
    
    decision = "Reject the null hypothesis"
    if p_value>signifLevel:
        decision = "DONT Reject the null hypothesis"
    
    print("Decision: {}, H0, at the {} level of significance.".format(decision,signifLevel))


def ANOVA1_is_contrast(c):
    '''
    A linear combination c = (c1, . . . , cI ) of the means 
    is called a contrast if sum(c1, . . . , cI) = 0.
    '''
    if sum(c) == 0:
        return True
    else: return False
    
def ANOVA1_is_orthogonal(group_sizes, c1, c2):
    '''
    Two contrasts constructed by c1, c2 ∈ C are said to be
    orthogonal if their dot product divided by group size at each point is zero.
    '''
    if not ANOVA1_is_contrast(c1) or not ANOVA1_is_contrast(c2):
        print("WARNING: One of the coefficient vectors is not a constrast!")
        return False
    
    elif sum([c1[i]*c2[i]/n for i,n in enumerate(group_sizes)]) == 0:
        return True

    else: return False
    

def Bonferroni_correction(FWER, numTests):
    #return significance level for each test with Bonferroni simply alpha/m
    return FWER / numTests

def Sidak_correction(FWER, numTests):
    #return significance level for each test with Sidak simply 1-(1-alpha)^(1/m)
    return 1-(1-FWER)**(1/numTests)

def Scheffe(data,C,n,I,J,SSw,signifLevel):
    intervals = []
    for c in C:
        squaredTerm = np.sqrt((SSw/(n-I))*sum((c[i]**2)/J[i] for i in range(len(c))))
        lhs = sum(c[i]*np.mean(data[i]) for i in range(len(c)))
     
        if ANOVA1_is_contrast(c):
            fstats = stats.f.ppf(q=1-signifLevel, dfn=(I-1), dfd=n-I)
            rhs = np.sqrt((I-1)*fstats)*squaredTerm
        else:
            fstats = stats.f.ppf(q=1-signifLevel, dfn=I, dfd=n-I)
            rhs = np.sqrt(I*fstats)*squaredTerm
        intervals.append((lhs-rhs, lhs+rhs))
    return intervals

def Scheffe_with_NOT_all_Constrast(data,C,n,I,J,SSw,signifLevel):
    intervals = []
    for c in C:
        squaredTerm = np.sqrt((SSw/(n-I))*sum((c[i]**2)/J[i] for i in range(len(c))))
        lhs = sum(c[i]*np.mean(data[i]) for i in range(len(c)))
        fstats = stats.f.ppf(q=1-signifLevel, dfn=I, dfd=n-I)
        rhs = np.sqrt(I*fstats)*squaredTerm
        intervals.append((lhs-rhs, lhs+rhs))
    return intervals

def Bonferroni(data,C,n,I,J,SSw,signifLevel):
    intervals = []
    for c in C:
        level = Bonferroni_correction(signifLevel, len(C))
        squaredTerm = np.sqrt((SSw/(n-I))*sum((c[i]**2)/J[i] for i in range(len(c))))
        lhs = sum(c[i]*np.mean(data[i]) for i in range(len(c)))
        tstats = stats.t.ppf(1-level, n-I)
        rhs = tstats*squaredTerm
        intervals.append((lhs-rhs, lhs+rhs))
    return intervals

def Sidak(data,C,n,I,J,SSw,signifLevel):
    intervals = []
    for c in C:
        level = Sidak_correction(signifLevel, len(C))
        squaredTerm = np.sqrt((SSw/(n-I))*sum((c[i]**2)/J[i] for i in range(len(c))))
        lhs = sum(c[i]*np.mean(data[i]) for i in range(len(c)))
        tstats = stats.t.ppf(1-level, n-I)
        rhs = tstats*squaredTerm
        intervals.append((lhs-rhs, lhs+rhs))
    return intervals

def Tukey(data,C,n,I,J,SSw,signifLevel):
    intervals = []
    for c in C:
        squaredTerm = np.sqrt((SSw/(n-I))*2/J[0])
        lhs = sum(c[i]*np.mean(data[i]) for i in range(len(c)))
        qstats = qsturng(1-signifLevel, I, n-I)
        rhs = (qstats/(np.sqrt(2)))*squaredTerm
        intervals.append((lhs-rhs, lhs+rhs))
    return intervals

def find_intersection(intervals):
    #returns the intersection of all intervals
    maxLower=10**8
    minUpper=-10**8
    for pair in intervals:
        if pair[0]<maxLower:
            maxLower=pair[0]
        if pair[1]>minUpper:
            minUpper=pair[0]
    return (maxLower,minUpper)

def IsPairwiseDiff(C):
    isValid=True
    for c in C:
        if np.count_nonzero(c) != 2 and not sum(c)==0:
            isValid =False
    return isValid

def ANOVA1_CI_linear_combs(data, C, method ="best", signifLevel = 0.05): #best and alpha=0.05 are default values
    #possible methods: "Bonferroni", "Tukey", "Scheffe", "Sidak" or "best": best of all
    I = len(data)
    J = [len(treatment) for treatment in data] 
    n = sum(J) #number of data points
    SStotal, SSw, SSb = ANOVA1_partition_TSS(data).values()
    isValid=True
    if method == "Tukey": #valid for pairwise comparisons
        #if all row of C contains 2 non-zero entry and sum==0--> pairwise comparisons
        if IsPairwiseDiff(C):
            isValid=False
    if isValid:
        CI = dict()
        
        if method == "Scheffe":
            CI["Scheffe"] = Scheffe(data,C,n,I,J,SSw,signifLevel)
            return CI["Scheffe"]
        elif method == "Bonferroni":
            CI["Bonferroni"] = Bonferroni(data,C,n,I,J,SSw,signifLevel)
            return CI["Bonferroni"]
        elif method == "Sidak":
            CI["Sidak"] = Sidak(data,C,n,I,J,SSw,signifLevel)       
            return CI["Sidak"]
        elif method=="Tukey":
            CI["Tukey"] = Tukey(data,C,n,I,J,SSw,signifLevel)
            return CI["Tukey"]

        else: #method is best
            isAllConstrast = True
            CI = dict()
            for c in C:
                if not ANOVA1_is_contrast(c):
                    isAllConstrast = False
            if isAllConstrast:
                #case 1 all constrast and all orthogonal compare compare Scheffe with Sidak’s correction.
                isAllOrthogonal = True
                for pair in list(combinations(C, 2)):
                    if not ANOVA1_is_orthogonal(J,pair[0],pair[1]):
                        isAllOrthogonal=False
                if isAllOrthogonal:
                    CI["Sidak"] = Sidak(data,C,n,I,J,SSw,signifLevel)
                    best1 = find_intersection(CI["Sidak"])
                    CI["Scheffe"] = Scheffe(data,C,n,I,J,SSw,signifLevel)
                    best2 = find_intersection(CI["Scheffe"])
                    if abs(best1[1]-best1[0])<abs(best2[1]-best2[0]):
                        CI["best"] = CI["Sidak"]
                    else:
                        CI["best"] = CI["Scheffe"]
                else:
                    CI["Bonferroni"] = Bonferroni(data,C,n,I,J,SSw,signifLevel)
                    best1 = find_intersection(CI["Bonferroni"])
                    CI["Scheffe"] = Scheffe(data,C,n,I,J,SSw,signifLevel)
                    best2 = find_intersection(CI["Scheffe"])
                    if abs(best1[1]-best1[0])<abs(best2[1]-best2[0]):
                        CI["best"] = CI["Bonferroni"]
                    else:
                        CI["best"] = CI["Scheffe"]
                
                
                if IsPairwiseDiff(C):
                    CI["Bonferroni"] = Bonferroni(data,C,n,I,J,SSw,signifLevel)
                    best1 = find_intersection(CI["Bonferroni"])
                    CI["Tukey"] = Scheffe(data,C,n,I,J,SSw,signifLevel)

                    best2 = find_intersection(CI["Tukey"])
                    if abs(best1[1]-best1[0])<abs(best2[1]-best2[0]):
                        CI["best"] = CI["Bonferroni"]
                    else:
                        CI["best"] = CI["Tukey"]
                    if isAllOrthogonal:
                        CI["Sidak"] = Sidak(data,C,n,I,J,SSw,signifLevel)
                        best3 = find_intersection(CI["Sidak"])
                        if abs(best2[1]-best2[0])<abs(best3[1]-best3[0]):
                            CI["best"] = CI["Tukey"]
                        else:
                            CI["best"] = CI["Sidak"]
            else:
                CI["Bonferroni"] = Bonferroni(data,C,n,I,J,SSw,signifLevel)
                best1 = find_intersection(CI["Bonferroni"])
                CI["Scheffe"] = Scheffe_with_NOT_all_Constrast(data,C,n,I,J,SSw,signifLevel)
                best2 = find_intersection(CI["Scheffe"])
                if abs(best1[1]-best1[0])<abs(best2[1]-best2[0]):
                    CI["best"] = CI["Bonferroni"]
                else:
                    CI["best"] = CI["Scheffe"]
            
            return CI["best"]
    else:
        print("Given method: Tukey's confidence intervals are valid for pairwise comparisons only.")
        return None

def ANOVA1_test_linear_combs(data,C,d,FWER=0.05,method="best"):
    isValid=True
    if method == "Tukey": #valid for pairwise comparisons
        #if all row of C contains 2 non-zero entry and sum==0--> pairwise comparisons
        if IsPairwiseDiff(C):
            isValid=False
    if isValid:
        confInterval = ANOVA1_CI_linear_combs(data,C,method,FWER)
        overallTest = True
        for i,di in enumerate(d):
            if di<=confInterval[i][1] and di>=confInterval[i][0]:
                print("d{} is inside the confidence interval. H0{} is NOT Rejected".format(i+1,i+1))
            else:
                print("d{} is NOT inside the confidence interval. H0{} is Rejected".format(i+1,i+1))
                overallTest=False
        if overallTest:
            return "H0 is NOT rejected"
        
        else:
            return "H0 is rejected"
    else:
        print("Tukey: Tukey's confidence intervals are valid for pairwise comparisons only.")
        return None

def ANOVA2_partition_TSS(data):
    '''
    This function partitions the sum of squares in a two way ANOVA layout.
    The function takes a data set Xi,j,k for i = 1, . . . ,I and j = 1, . . . , J and k = 1, . . . ,K and return
    SStotal, SSa, SSb, SSab, SSe

    SStotal = SSa + SSb + SSab + SSe.

    '''
    I = len(data)
    J = len(data[0])
    K = len(data[0][0])
    meanAll = np.mean([k for j in [j for i in data for j in i] for k in j]) #linearize the data then get mean
    meansI = [np.hstack(i).mean() for i in data]
    meansJ = [np.mean(j) for j in zip(*[i for i in data])]
    meansIJ = [[np.mean(j) for j in i] for i in data]
    
    SSa = J*K*sum((meansI[i]-meanAll)**2 for i in range(I))
    SSb = I*K*sum((meansJ[j]-meanAll)**2 for j in range(J))
    SSab = K*sum(sum((meansIJ[i][j]-meansJ[j]-meansI[i]+meanAll)**2 for j in range(J)) for i in range(I))
    SSe = sum(sum(sum((data[i][j][k]-meansIJ[i][j])**2 for k in range(K)) for j in range(J)) for i in range(I))
    
    SStotal = SSa + SSb + SSab + SSe
    
    return {"SSa": SSa,"SSb": SSb,"SSab": SSab,"SSe": SSe,"SStotal":SStotal}


def ANOVA2_MLE(data):
    #returns MLE estimates of µ,ai,bj and δij
    I = len(data)
    J = len(data[0])
    K = len(data[0][0])
    meanAll = np.mean([k for j in [j for i in data for j in i] for k in j]) #linearize the data then get mean
    meansI = [np.hstack(i).mean() for i in data]
    meansJ = [np.mean(j) for j in zip(*[i for i in data])]
    meansIJ = [[np.mean(j) for j in i] for i in data]
    
    return {"µ":meanAll, "ai":[meansI[i]-meanAll for i in range(I)], "bj":[meansJ[j]-meanAll for j in range(J)], "δij":[[meansIJ[i][j]-meansJ[j]-meansI[i]+meanAll for j in range(J)] for i in range(I)]}
    
def printANOVATable(data):
    I = len(data)
    J = len(data[0])
    K = len(data[0][0])

    table = {"degrees of freedom" : [I-1,J-1,(I-1)*(J-1),(I*J*(K-1)),I*J*K-1]}
    table["SS"]= list(ANOVA2_partition_TSS(data).values())
    MS = [x/y for x, y in zip(table["SS"][:-1], table["degrees of freedom"][:-1])]
    MS.append("")
    table["MS"] = MS #except the total
    
    table["F"]= [table["MS"][0]/table["MS"][3],table["MS"][1]/table["MS"][3],table["MS"][2]/table["MS"][3],"",""]
    tabledf = pd.DataFrame(table,index=["A","B","AxB","within","Total"])
    print(tabledf)
    return table

def DecisionANOVA(table, choice,index,alpha):
    fstats = stats.f.ppf(q=1-alpha, dfn=table["degrees of freedom"][index], dfd=table["degrees of freedom"][3])
    print("Critical value is: ",fstats)
    print("Compared F-value is: ",table["F"][index])
    return fstats<table["F"][index] #return false if rejected

def ANOVA2_test_equality(data, alpha = 0.05, choice = "A"):
    table =printANOVATable(data)
    print("-------------------------------")
    if choice=="B":
        if not DecisionANOVA(table, choice,1,alpha):
            print("The hypothesis b1=b2=b3=...bI=0 is {} Rejected".format("NOT"))
        else:
            print("The hypothesis b1=b2=b3=...bI=0 is Rejected")
    elif choice =="AB":
        if not DecisionANOVA(table, choice,2,alpha):
            print("The hypothesis that δij=0 forall i,j is {} Rejected".format("NOT"))
        else:
            print("The hypothesis that δij=0 forall i,j is Rejected")
    else:
        if not DecisionANOVA(table, choice,0,alpha):
            print("The hypothesis a1=a2=a3=...aI=0 is {} Rejected".format("NOT"))
        else:
            print("The hypothesis a1=a2=a3=...aI=0 is Rejected")

    