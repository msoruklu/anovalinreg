from anova import Bonferroni_correction
import numpy as np
from pprint import pprint
#stats for find t,f values at given alpha and dofs
from scipy import stats

def Mult_LR_Least_squares(X,y):
    '''
    Function that finds the least squares solution according to the multiple linear
    regression model: The function takes X and y, the design matrix and the response
    vector, and produces the maximum likelihood estimators for β and σ_square as well as the
    unbiased estimate for σ_square
    '''
    #design is X matrix and response is y
    beta=np.dot(np.dot(np.linalg.inv(np.dot(np.array(X).transpose(),np.array(X))),np.array(X).transpose()),y)
    predictions = np.dot(np.array(X),beta)
    # print(predictions,y)
    error = predictions-y
    sigmaSqr = sum(np.square(error))/len(X)
    SeSqr = sum(np.square(error))/(len(X)-len(X[0]))  #RSS/n-k-1
    return {"beta_hat":beta,"sigma_hatSqr":sigmaSqr,"SeSqr":SeSqr}

def Mult_LR_partition_TSS(X,y):
    '''
    Function that takes an n × (k + 1) matrix X, n × 1 vector y as inputs and
    returns the total sum of squares, regression sum of squares, and residual sum of
    squares.
    '''
    ybar = np.mean(y)
    SStotal = sum((i-ybar)**2 for i in y)
    beta = Mult_LR_Least_squares(X,y)["beta_hat"]
    preds = np.matmul(np.array(X),beta)
    RSS = sum((y[i]-preds[i])**2 for i,val in enumerate(y))
    RegSS = SStotal-RSS
    return {"SStotal": SStotal,"RegSS": RegSS, "RSS": RSS}

def Mult_norm_LR_simul_CI(X,y,alpha=0.05):
    '''
    Function that takes X and y, the design matrix and the response vector, and
    a significance parameter α, and produces confidence intervals for βi’s 
    that simultaneously hold with probability 1 − α.
    '''
    n=len(X)
    k = len(X[0])-1
    beta,sigmaSqr,SeSqr = Mult_LR_Least_squares(X,y).values()
    Se = np.sqrt(SeSqr)
    fstats = stats.f.ppf(1-alpha, k+1, n-k-1)
    XTXinv = np.linalg.inv(np.matmul(np.array(X).transpose(),np.array(X)))
    CI=[]
    for i,bi in enumerate(beta):
        rhs = np.sqrt((k+1)*fstats)*Se*np.sqrt(XTXinv[i][i])
        lower = bi-rhs
        upper = bi+rhs
        CI.append({"beta_{}".format(i):{"lower":lower,"upper":upper}})
    return CI
    
def Mult_norm_LR_CR(X,y,C,alpha=0.05):
    '''
    Function that takes an n×(k+1) matrix X, n×1 vector y, an r×(k+1) matrix
    C with rank r, and a significance level α as inputs, and returns the specifications
    (that is, parameters of the ellipsoid) of the 100(1 − α)% confidence region for Cβ
    according to the normal multiple linear regression model.
    '''
    #returns the parameters of ellipsoid ( rhs, C*B_hat, C*(XT*X)−1*CT ) 
    X,y,C = np.array(X),np.array(y),np.array(C)
    if (len(C.shape)) == 1: r=1
    else: r= C.shape[0]
    n=len(X)
    k = len(X[0])-1
    beta_hat,sigmaSqr,SeSqr = Mult_LR_Least_squares(X,y).values()
    fstats = stats.f.ppf(1-alpha, r, n-k-1)
    rhs = r*SeSqr*fstats
    CB_hat = np.matmul(C,beta_hat)
    XTXinv = np.linalg.inv(np.matmul(X.transpose(),X))
    if r==1: C_XtX_Ct_inv = 1/np.matmul(np.matmul(C,XTXinv),C.transpose())
    else: C_XtX_Ct_inv = np.linalg.inv(np.matmul(np.matmul(C,XTXinv),C.transpose()))

    return {"CB_hat":CB_hat, "C_XtX_Ct_inv":C_XtX_Ct_inv, "rhs":rhs}

def Mult_norm_LR_is_in_CR(X,y,C,c_zero,alpha=0.05):
    '''
    Function that takes an n × (k + 1) matrix X, n × 1 vector y, an r × (k + 1)
    matrix C with rank r, a r × 1 vector c0, and a significance level α as inputs, and
    answers whether c0 is in the 100(1 − α)% confidence region for Cβ according to the
    normal multiple linear regression model.
    '''
    c_zero = np.array(c_zero)
    CB_hat, C_XtX_Ct_inv, rhs = Mult_norm_LR_CR(X,y,C,alpha).values()
    CB_hat_c_zero = CB_hat-c_zero
    lhs = np.dot(np.dot(CB_hat_c_zero.transpose(),C_XtX_Ct_inv),CB_hat_c_zero)


    #return False if c_zero is inside of critical reigon
    if lhs<=rhs:
        return True
    else:
        return False

def Mult_norm_LR_test_general(X,y,C,c_zero,alpha=0.05):
    '''Function that takes an n × (k + 1) matrix X, n × 1 vector y, an r × (k + 1)
        matrix C with rank r, a r×1 vector c0, and a significance level α as inputs, and tests
        the null hypothesis H0 : Cβ = c0 vs H1 : Cβ != c0 at a significance level of α. '''

    decision = Mult_norm_LR_is_in_CR(X,y,C,c_zero,alpha)

    if not decision:
        print("{} is NOT in the {}% confidence region. REJECT H0!".format(list(c_zero),(1-alpha)*100))
    else:
        print("{} is in the {}% confidence region. DONT REJECT H0!".format(list(c_zero),(1-alpha)*100))

def Mult_norm_LR_test_comp(X,y,J,alpha=0.05):
    '''Function that takes an n × (k + 1) matrix X, n × 1 vector y, a significance
    level α, and j1, . . . , jr ∈ {0, . . . , k} as inputs, and returns the outcome of testing
    H0 : βj1 = . . . = βjr = 0 vs H1 : not H0.'''

    X,y,J = np.array(X),np.array(y),np.array(J)
    C = np.zeros(shape =(len(J),len(X[0])))
    
    #create a suitable C from J
    for i in range(len(J)):
        C[i][J[i]]=1

    c_zero = np.zeros(len(J))

    Mult_norm_LR_test_general(X,y,C,c_zero,alpha)

def Mult_norm_LR_test_linear_reg(X,y,alpha=0.05):
    J=np.arange(1,len(X[0]),1)

    Mult_norm_LR_test_comp(X,y,J,alpha)

def Mult_norm_LR_simul_CI_Bonferroni(X,y,D,alpha=0.05):
    #function to calculate confidence intervals according do Bonferroni correction
    n=len(X)
    k = len(X[0])-1
    beta,sigmaSqr,SeSqr = Mult_LR_Least_squares(X,y).values()
    Se = np.sqrt(SeSqr)
    tstats = stats.t.ppf(1-alpha, n-k-1) 
    XTXinv = np.linalg.inv(np.matmul(np.array(X).transpose(),np.array(X)))
    CI=[]
    for i,di in enumerate(D):
        sqr_term = np.dot(np.dot(np.transpose(di),XTXinv),di)
        rhs = tstats*Se*np.sqrt(sqr_term)
        lower = np.dot(np.transpose(di),beta)-rhs
        upper = np.dot(np.transpose(di),beta)+rhs
        CI.append([lower,upper])
    return CI
    
def Mult_norm_LR_simul_CI_Scheffe(X,y,D,alpha=0.05):
    #function to calculate confidence intervals according do Scheffe
    n=len(X)
    k = len(X[0])-1
    beta,sigmaSqr,SeSqr = Mult_LR_Least_squares(X,y).values()
    Se = np.sqrt(SeSqr)
    fstats = stats.f.ppf(1-alpha,k+1, n-k-1) 
    XTXinv = np.linalg.inv(np.matmul(np.array(X).transpose(),np.array(X)))
    CI=[]
    for i,di in enumerate(D):
        sqr_term1 = np.dot(np.dot(np.transpose(di),XTXinv),di)
        sqr_term2 = (k+1)*fstats
        rhs = np.sqrt(sqr_term2)*Se*np.sqrt(sqr_term1)
        lower = np.dot(np.transpose(di),beta)-rhs
        upper = np.dot(np.transpose(di),beta)+rhs
        CI.append([lower,upper])
    return CI

def Mult_norm_LR_pred_CI(X,y,D,alpha=0.05,method="best"):
    '''Function that takes a n × (k + 1) matrix X, n × 1 vector y, a m × (k + 1)
        matrix D, a significance level α, and a method as inputs, and returns simultaneous
        confidence bounds for diβ for all i = 1, . . . , m according to the normal multiple linear
        regression model, where di is the i’th row of the matrix.'''
    #we use Mult_norm_LR_simul_CI and Mult_norm_LR_simul_CI_Bonferroni functions here and return narrower one if method is "best"
    bonferroni_level = Bonferroni_correction(alpha,len(D))
    CI_bonfer=Mult_norm_LR_simul_CI_Bonferroni(X,y,D,bonferroni_level)
    CI_Scheffe = Mult_norm_LR_simul_CI_Scheffe(X,y,D,alpha)

    if method=="Bonferroni":
        return CI_bonfer
    elif method=="Scheffe":
        return CI_Scheffe
    #check the best, narrower is the best
    else:
        minBonfer = min([CI_bonfer[i][1]-CI_bonfer[i][0] for i in range(len(CI_bonfer))])
        minScheffe = min([CI_Scheffe[i][1]-CI_Scheffe[i][0] for i in range(len(CI_Scheffe))])
    
        if minBonfer<minScheffe: return {"Best is Bonferroni":CI_bonfer} 
        else: return {"Best is Scheffe":CI_Scheffe}
 






