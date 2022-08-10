# Support Vector Machine

``` r
# Load libraries
library(e1071)
library(caret)
```

    ## Loading required package: ggplot2

    ## Loading required package: lattice

## Import Data

``` r
# Read training and testing data
train <- read.csv("./../data/classification_data/intermediates/train.csv")
test <- read.csv("./../data/classification_data/intermediates/test.csv")
```

## Model Training

``` r
svm.model <- svm(as.factor(Rating)~., data=train, type='C-classification')
summary(svm.model)
```

    ## 
    ## Call:
    ## svm(formula = as.factor(Rating) ~ ., data = train, type = "C-classification")
    ## 
    ## 
    ## Parameters:
    ##    SVM-Type:  C-classification 
    ##  SVM-Kernel:  radial 
    ##        cost:  1 
    ## 
    ## Number of Support Vectors:  6108
    ## 
    ##  ( 563 547 197 641 648 726 367 243 326 72 348 413 229 374 53 154 121 15 9 9 53 )
    ## 
    ## 
    ## Number of Classes:  21 
    ## 
    ## Levels: 
    ##  1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21

## Model Validation

``` r
result = predict(svm.model, test, type="raw")

confusionMatrix(as.factor(result), as.factor(test$Rating))
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21
    ##         1   4  0  1  2  0  1  2  0  1  0  0  0  0  0  0  0  0  0  0  0  0
    ##         2   0  7  0  0  0  0  2  1  0  0  0  0  0  0  0  0  0  0  0  0  0
    ##         3   0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
    ##         4   0  0  0  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
    ##         5   0  4  6  8 37 13 11  5  5  5  0  0  0  0  0  0  0  0  0  0  0
    ##         6   6  2 13  9 24 75 35 38 18 17  0  0  0  0  0  0  0  0  0  0  0
    ##         7   0  0  2 22 27 31 40 33 26 13  0  0  0  0  0  0  0  0  0  0  0
    ##         8   3  4  4  5  0 17 20 39 30 21  0  0  0  0  0  0  0  0  0  0  0
    ##         9   2  0  4  0 10 29 14 51 71 47  0  0  0  0  0  0  0  0  0  0  0
    ##         10  1  0  1  1  1  9  9 11 19 40  0  0  0  0  0  0  0  0  0  0  0
    ##         11  0  0  0  0  0  0  0  0  0  0 59 36 12 13 15 10  4  1  0  0  2
    ##         12  0  0  0  0  0  0  0  0  0  0 27 26 15 11  9  7  6  1  0  2  0
    ##         13  0  0  0  0  0  0  0  0  0  0 11 14 28 18 10 10  8  4  0  0  0
    ##         14  0  0  0  0  0  0  0  0  0  0  9  6 14 33 14 10  7  0  0  0  0
    ##         15  0  0  0  0  0  0  0  0  0  0  3  1  2 12 12  7  1  1  1  0  0
    ##         16  0  0  0  0  0  0  0  1  0  0  2  2  8  5  5 14  2  3  0  0  0
    ##         17  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1  0  0  1  0
    ##         18  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1  0  0  0
    ##         19  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
    ##         20  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
    ##         21  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.3154          
    ##                  95% CI : (0.2923, 0.3393)
    ##     No Information Rate : 0.1157          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.2523          
    ##                                           
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: 1 Class: 2 Class: 3  Class: 4 Class: 5 Class: 6
    ## Sensitivity          0.250000 0.411765  0.00000 0.0208333  0.37374  0.42857
    ## Specificity          0.995428 0.998039  1.00000 1.0000000  0.96064  0.88192
    ## Pos Pred Value       0.363636 0.700000      NaN 1.0000000  0.39362  0.31646
    ## Neg Pred Value       0.992188 0.993494  0.97996 0.9695990  0.95733  0.92366
    ## Prevalence           0.010343 0.010989  0.02004 0.0310278  0.06399  0.11312
    ## Detection Rate       0.002586 0.004525  0.00000 0.0006464  0.02392  0.04848
    ## Detection Prevalence 0.007111 0.006464  0.00000 0.0006464  0.06076  0.15320
    ## Balanced Accuracy    0.622714 0.704902  0.50000 0.5104167  0.66719  0.65525
    ##                      Class: 7 Class: 8 Class: 9 Class: 10 Class: 11 Class: 12
    ## Sensitivity           0.30075  0.21788   0.4176   0.27972   0.53153   0.30588
    ## Specificity           0.89109  0.92398   0.8860   0.96296   0.93524   0.94665
    ## Pos Pred Value        0.20619  0.27273   0.3114   0.43478   0.38816   0.25000
    ## Neg Pred Value        0.93126  0.90028   0.9249   0.92921   0.96272   0.95911
    ## Prevalence            0.08597  0.11571   0.1099   0.09244   0.07175   0.05495
    ## Detection Rate        0.02586  0.02521   0.0459   0.02586   0.03814   0.01681
    ## Detection Prevalence  0.12540  0.09244   0.1474   0.05947   0.09825   0.06723
    ## Balanced Accuracy     0.59592  0.57093   0.6518   0.62134   0.73338   0.62627
    ##                      Class: 13 Class: 14 Class: 15 Class: 16 Class: 17
    ## Sensitivity            0.35443   0.35870  0.184615   0.24138 0.0344828
    ## Specificity            0.94891   0.95876  0.981107   0.98120 0.9993412
    ## Pos Pred Value         0.27184   0.35484  0.300000   0.33333 0.5000000
    ## Neg Pred Value         0.96468   0.95942  0.964831   0.97076 0.9818770
    ## Prevalence             0.05107   0.05947  0.042017   0.03749 0.0187460
    ## Detection Rate         0.01810   0.02133  0.007757   0.00905 0.0006464
    ## Detection Prevalence   0.06658   0.06012  0.025856   0.02715 0.0012928
    ## Balanced Accuracy      0.65167   0.65873  0.582861   0.61129 0.5169120
    ##                      Class: 18 Class: 19 Class: 20 Class: 21
    ## Sensitivity          0.0909091 0.0000000  0.000000  0.000000
    ## Specificity          1.0000000 1.0000000  1.000000  1.000000
    ## Pos Pred Value       1.0000000       NaN       NaN       NaN
    ## Neg Pred Value       0.9935317 0.9993536  0.998061  0.998707
    ## Prevalence           0.0071105 0.0006464  0.001939  0.001293
    ## Detection Rate       0.0006464 0.0000000  0.000000  0.000000
    ## Detection Prevalence 0.0006464 0.0000000  0.000000  0.000000
    ## Balanced Accuracy    0.5454545 0.5000000  0.500000  0.500000
