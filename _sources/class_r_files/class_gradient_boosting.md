# Gradient Boosting

``` r
# Load libraries
library(gbm)
```

    ## Loaded gbm 2.1.8

``` r
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
gbm.model = gbm(as.factor(Rating)~., data=train, distribution = "multinomial")
```

    ## Warning: Setting `distribution = "multinomial"` is ill-advised as it is
    ## currently broken. It exists only for backwards compatibility. Use at your own
    ## risk.

``` r
gbm.model
```

    ## gbm(formula = as.factor(Rating) ~ ., distribution = "multinomial", 
    ##     data = train)
    ## A gradient boosted model with multinomial loss function.
    ## 100 iterations were performed.
    ## There were 31 predictors of which 30 had non-zero influence.

``` r
result <- predict(gbm.model, test)
```

    ## Using 100 trees...

``` r
final.result = colnames(result)[apply(result,1,which.max)]

confusionMatrix(as.factor(final.result), as.factor(test$Rating))
```

    ## Warning in levels(reference) != levels(data): longer object length is not a
    ## multiple of shorter object length

    ## Warning in confusionMatrix.default(as.factor(final.result),
    ## as.factor(test$Rating)): Levels are not in the same order for reference and
    ## data. Refactoring data to match.

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21
    ##         1   8  0  0  0  0  3  3  0  0  2  0  0  0  0  0  0  0  0  0  0  0
    ##         2   1  7  0  0  0  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
    ##         3   0  0  5  1  6  1  0  1  0  1  0  0  0  0  0  0  0  0  0  0  0
    ##         4   0  4  1 14  7  0  3  2  1  0  0  0  0  0  0  1  0  0  0  0  0
    ##         5   0  1 11 15 32  9 18 10  4  2  0  0  0  0  0  0  0  0  0  0  0
    ##         6   2  0  7  6 16 53 21 30 17 19  0  0  0  0  0  0  0  0  0  0  0
    ##         7   1  0  1  9 22 30 32 28  7 10  0  0  0  0  0  0  0  0  0  0  0
    ##         8   2  3  2  0  3 22 27 46 27 14  0  0  0  0  0  0  0  0  0  0  0
    ##         9   2  2  4  2 12 44 24 50 91 52  0  0  0  0  0  0  0  0  0  0  0
    ##         10  0  0  0  1  1 12  5 12 23 43  0  0  0  0  0  0  0  0  0  0  0
    ##         11  0  0  0  0  0  0  0  0  0  0 62 36 18 25 11  7  3  6  0  0  1
    ##         12  0  0  0  0  0  0  0  0  0  0 20 27 20 13  7 10  5  1  0  2  0
    ##         13  0  0  0  0  0  0  0  0  0  0 14 15 19 14 11 10  3  0  0  0  0
    ##         14  0  0  0  0  0  0  0  0  0  0 11  2 12 26 11  7  2  0  0  0  0
    ##         15  0  0  0  0  0  0  0  0  0  0  2  2  2  4 13  4  6  1  1  0  0
    ##         16  0  0  0  0  0  0  0  0  0  0  2  3  8  9 10 17  4  2  0  0  1
    ##         17  0  0  0  0  0  0  0  0  0  0  0  0  0  1  0  1  2  0  0  1  0
    ##         18  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1  1  4  1  0  0  0
    ##         19  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1  0  0  0  0  0  0
    ##         20  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
    ##         21  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.3219          
    ##                  95% CI : (0.2987, 0.3458)
    ##     No Information Rate : 0.1157          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.2614          
    ##                                           
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: 1 Class: 2 Class: 3 Class: 4 Class: 5 Class: 6
    ## Sensitivity          0.500000 0.411765 0.161290  0.29167  0.32323  0.30286
    ## Specificity          0.994775 0.998693 0.993404  0.98732  0.95166  0.91399
    ## Pos Pred Value       0.500000 0.777778 0.333333  0.42424  0.31373  0.30994
    ## Neg Pred Value       0.994775 0.993498 0.983029  0.97754  0.95363  0.91134
    ## Prevalence           0.010343 0.010989 0.020039  0.03103  0.06399  0.11312
    ## Detection Rate       0.005171 0.004525 0.003232  0.00905  0.02069  0.03426
    ## Detection Prevalence 0.010343 0.005818 0.009696  0.02133  0.06593  0.11054
    ## Balanced Accuracy    0.747387 0.705229 0.577347  0.63950  0.63744  0.60843
    ##                      Class: 7 Class: 8 Class: 9 Class: 10 Class: 11 Class: 12
    ## Sensitivity           0.24060  0.25698  0.53529   0.30070   0.55856   0.31765
    ## Specificity           0.92362  0.92690  0.86057   0.96154   0.92549   0.94665
    ## Pos Pred Value        0.22857  0.31507  0.32155   0.44330   0.36686   0.25714
    ## Neg Pred Value        0.92822  0.90507  0.93750   0.93103   0.96444   0.95978
    ## Prevalence            0.08597  0.11571  0.10989   0.09244   0.07175   0.05495
    ## Detection Rate        0.02069  0.02973  0.05882   0.02780   0.04008   0.01745
    ## Detection Prevalence  0.09050  0.09438  0.18293   0.06270   0.10924   0.06787
    ## Balanced Accuracy     0.58211  0.59194  0.69793   0.63112   0.74202   0.63215
    ##                      Class: 13 Class: 14 Class: 15 Class: 16 Class: 17
    ## Sensitivity            0.24051   0.28261  0.200000   0.29310  0.068966
    ## Specificity            0.95436   0.96907  0.985155   0.97381  0.998024
    ## Pos Pred Value         0.22093   0.36620  0.371429   0.30357  0.400000
    ## Neg Pred Value         0.95893   0.95528  0.965608   0.97250  0.982490
    ## Prevalence             0.05107   0.05947  0.042017   0.03749  0.018746
    ## Detection Rate         0.01228   0.01681  0.008403   0.01099  0.001293
    ## Detection Prevalence   0.05559   0.04590  0.022624   0.03620  0.003232
    ## Balanced Accuracy      0.59743   0.62584  0.592578   0.63346  0.533495
    ##                      Class: 18 Class: 19 Class: 20 Class: 21
    ## Sensitivity          0.0909091 0.0000000  0.000000  0.000000
    ## Specificity          0.9960938 0.9993532  1.000000  1.000000
    ## Pos Pred Value       0.1428571 0.0000000       NaN       NaN
    ## Neg Pred Value       0.9935065 0.9993532  0.998061  0.998707
    ## Prevalence           0.0071105 0.0006464  0.001939  0.001293
    ## Detection Rate       0.0006464 0.0000000  0.000000  0.000000
    ## Detection Prevalence 0.0045249 0.0006464  0.000000  0.000000
    ## Balanced Accuracy    0.5435014 0.4996766  0.500000  0.500000
