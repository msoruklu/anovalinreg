# Toolbox-for-ANOVA-and-Linear-Regression
This project  is a toolbox for ANOVA and linear regression, which is package of functions. 
The package contains the functions named and described below.

## ANOVA 

* ANOVA1_partition_TSS(data)
    This function partitions the sum of squares in a one way ANOVA layout. The function takes a data set Xi,j for j = 1, . . . ,ni and i = 1, . . . , I, and return SStotal, SSw, and SSb. SStotal = SSw + SSb.

* ANOVA1_test_equality(data,signifLevel = 0.05):
 
    This function tests the equality of the means in a one way ANOVA layout. The input is a data set Xi,j for j = 1, . . . , ni and i = 1, . . . , I, and the significance level α.(default=0.05) As the output, the function prints all the quantities in the table as well as the critical value, the p-value, and the decision.

* ANOVA1_is_contrast(c)
    Returns true if a gicen vector c is a contrast. A linear combination c = (c1, . . . , cI ) of the means 
    is called a contrast if sum(c1, . . . , cI) = 0.

* ANOVA1_is_orthogonal(group_sizes, c1, c2)
    Returns true if given vectors are orthogonal. Two contrasts constructed by c1, c2 ∈ C are said to be
    orthogonal if their dot product divided by group size at each point is zero.

* Bonferroni_correction(FWER, numTests):
    Returns significance level for each test with Bonferroni correction.

* Sidak_correction(FWER, numTests):
    Returns significance level for each test with Sidak's correction.

* ANOVA1_CI_linear_combs(data, C, method ="best", signifLevel = 0.05)
    Input:
        – a data set Xi,j for j = 1, . . . , ni and i = 1, . . . , I,
        – Significance level α,
        – An m × I matrix C, where each row defines linear combination of the group means.
        – Method: This may be “Scheffe”, “Tukey”, “Bonferroni”, “Sidak”, “best”
    Output: As the output, the function returns simultaneous confidence intervals for those linear combinations.

* ANOVA1_test_linear_combs(data,C,d,FWER=0.05,method="best")
    Input:
    – a data set Xi,j for j = 1, . . . , ni and i = 1, . . . , I,
    – FWER α,
    – An m × I matrix C and a m × 1 vector d, where each row of C defines linear combination of the group means and each element of d is the hypothesized value for the corresponding combination.
    – Method: This may be “Scheffe”, “Tukey”, “Bonferroni”, “Sidak”, “best”.
        H0 : c_i_1µ_1 + . . . + c_i_Iµ_I = d_i, i = 1, . . . , m.
    Output: As the output, the function returns the test outcomes, in such a way that FWER is kept at α.

* ANOVA2_partition_TSS(data):
    This function partitions the sum of squares in a two way ANOVA layout.
    The function takes a data set Xi,j,k for i = 1, . . . ,I and j = 1, . . . , J and k = 1, . . . ,K and return SStotal, SSa, SSb, SSab, SSe: SStotal = SSa + SSb + SSab + SSe.

* ANOVA2_MLE(data)
    This function returns the MLE estimates of µ,ai,bj and δij

* ANOVA2_test_equality(data, alpha = 0.05, choice = "A")
    This function performs one of the basic three tests in the two-way ANOVA layout. The function takes 
    Xi,j,k for i = 1, . . . , I, j = 1, . . . , J, and k = 1, . . . , K, and a significance level α and performs one of the following (depending on the choice).
        – The hypothesis that a1 = . . . = aI = 0.
        – The hypothesis that b1 = . . . = bI = 0.
        – The hypothesis that all δij ’s are equal to 0.
    The choice for the test should also be inputted as an input as either “A”, “B”, or
    “AB”. The function prints the ANOVA table. 


## LINEAR REGRESSION

* Mult_LR_Least_squares(X,y):
    Function that finds the least squares solution according to the multiple linear regression model: The function takes X and y, the design matrix and the response vector, and produces the maximum likelihood estimators for β and σ_square as well as th unbiased estimate for σ_square. 

* Mult_LR_partition_TSS(X,y):
    Function that takes an n × (k + 1) matrix X, n × 1 vector y as inputs and returns the total sum of squares, regression sum of squares, and residual sum of squares.

* Mult_norm_LR_simul_CI(X,y,alpha=0.05):
    Function that takes X and y, the design matrix and the response vector, and a significance parameter α, and produces confidence intervals for βi’s that simultaneously hold with probability 1 − α.

* Mult_norm_LR_CR(X,y,C,alpha=0.05):
    Function that takes an n×(k+1) matrix X, n×1 vector y, an r×(k+1) matrix C with rank r, and a significance level α as inputs, and returns the specifications (that is, parameters of the ellipsoid) of the 100(1 − α)% confidence region for Cβ according to the normal multiple linear regression model.

* Mult_norm_LR_is_in_CR(X,y,C,c_zero,alpha=0.05):
    Function that takes an n × (k + 1) matrix X, n × 1 vector y, an r × (k + 1) matrix C with rank r, a r × 1 vector c0, and a significance level α as inputs, and answers whether c0 is in the 100(1 − α)% confidence region for Cβ according to the normal multiple linear regression model.

* Mult_norm_LR_test_general(X,y,C,c_zero,alpha=0.05):
    Function that takes an n × (k + 1) matrix X, n × 1 vector y, an r × (k + 1) matrix C with rank r, a r×1 vector c0, and a significance level α as inputs, and tests the null hypothesis 
    H0 : Cβ = c0 vs H1 : Cβ 6= c0 at a significance level of α.

* Mult_norm_LR_test_comp(X,y,J,alpha=0.05):
    Function that takes an n × (k + 1) matrix X, n × 1 vector y, a significance level α, and 
    j1, . . . , jr ∈ {0, . . . , k} as inputs, and returns the outcome of testing 
        H0 : βj1 = . . . = βjr = 0 vs H1 : not H0.

* Mult_norm_LR_test_linear_reg(X,y,alpha=0.05)
    Function that takes an n × (k + 1) matrix X, n × 1 vector y, a significance level α as inputs, and returns the outcome of testing the existence of linear regressionat all, i.e., 
        H0 : β1 = . . . = βk = 0 vs H1 : not H0

* Mult_norm_LR_pred_CI(X,y,D,alpha=0.05,method="best"):
    Function that takes a n × (k + 1) matrix X, n × 1 vector y, a m × (k + 1) matrix D, a significance level α, and a method as inputs, and returns simultaneous confidence bounds for diβ for all i = 1, . . . , m according to the normal multiple linear regression model, where di is the i’th row of the matrix.

