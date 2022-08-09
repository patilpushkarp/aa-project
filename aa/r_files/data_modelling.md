# Data Modelling

Since the data has been cleaned, it can now be used to create the
models.

``` r
# Load libraries
library(caret)
```

    ## Loading required package: ggplot2

    ## Loading required package: lattice

``` r
library(nnet)
```

Since not many outliers are removed, there is very little information
loss and hence the data without outliers can be used.

``` r
# Load the data
df <- read.csv("./../data/classification_data/intermediates/preprocessed_data_without_outliers.csv")
```

## Partition the data

To evaluate the model, there should be a set of which the model has not
seen and for which the labels are known. Hence, it is necessary to split
the data into training and testing set.

``` r
# Partitioning the data
partition = createDataPartition(df$Rating, p=0.8, list = FALSE)
train = df[partition,]
test = df[-partition,]
```

## Logistic Regression

The model is first trained on the training data and then evaluated on
testing data.

``` r
# Model training
multinom.model <- multinom(Rating~., data=train, )
```

    ## # weights:  672 (620 variable)
    ## initial  value 18869.950069 
    ## iter  10 value 17309.913093
    ## iter  20 value 17274.304088
    ## iter  30 value 16887.203052
    ## iter  40 value 16729.133400
    ## iter  50 value 16636.629926
    ## iter  60 value 16592.512085
    ## iter  70 value 16453.536962
    ## iter  80 value 16029.229506
    ## iter  90 value 15359.556547
    ## iter 100 value 14842.206545
    ## final  value 14842.206545 
    ## stopped after 100 iterations

``` r
# Predict the samples from test data using the model
result <- predict(multinom.model, test)

# Print the Confusion matrix
confusionMatrix(as.factor(result), as.factor(test$Rating))
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21
    ##         1   0  0  0  0  0  0  0  0  0  1  4  0  0  0  1  0  0  0  0  0  0
    ##         2   0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
    ##         3   0  0  0  0  0  0  0  0  0  1  0  0  2  1  0  0  1  0  1  0  0
    ##         4   0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
    ##         5   0 10  5  8 20 10 16 10  3  1  1  0  0  2  0  1  0  0  0  0  0
    ##         6  16  5 13 10 12 75 33 51 48 28  4  2  0  0  1  0  0  0  0  0  0
    ##         7   2  0  4 17 20 21 31 14 16 13  0  1  1  1  1  0  0  0  0  0  0
    ##         8   2  1  5  5  8 15 20 27 19 19  0  1  0  2  0  0  0  0  0  0  0
    ##         9   0  1  9 17 23 41 32 61 78 73  0  0  0  0  0  0  0  0  0  0  0
    ##         10  0  0  0  0  0  0  1  3  8  4  0  0  0  0  0  0  0  0  0  0  0
    ##         11  0  0  1  0  0  1  0  2  1  1 64 50 32 34 18 33  7  6  0  0  1
    ##         12  0  0  1  0  0  0  0  0  0  0  4  2  2  1  0  0  1  1  0  0  0
    ##         13  0  0  0  0  0  0  0  0  0  0  5 13 13 10  4  1  3  0  0  0  0
    ##         14  0  0  0  0  0  0  0  0  0  0 16 16 37 27 22 20 10  3  1  0  0
    ##         15  0  0  0  0  2  0  3  2  1  0  2  6  1  1  7  5  1  2  0  1  0
    ##         16  0  0  0  0  0  0  1  2  0  0  1  1  3  4  0  1  2  0  0  1  0
    ##         17  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1  0  0  0  0
    ##         18  0  1  0  0  0  1  0  1  1  3  1  1  0  1  3  0  1  0  0  0  0
    ##         19  0  0  0  0  0  0  0  0  1  0  0  0  0  0  0  0  0  0  1  1  0
    ##         20  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
    ##         21  0  0  0  0  0  0  0  0  0  0  0  0  1  0  0  0  0  0  0  0  0
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.2269          
    ##                  95% CI : (0.2062, 0.2486)
    ##     No Information Rate : 0.1138          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.1541          
    ##                                           
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: 1 Class: 2 Class: 3 Class: 4 Class: 5 Class: 6
    ## Sensitivity          0.000000  0.00000 0.000000  0.00000  0.23529  0.45732
    ## Specificity          0.996071  1.00000 0.996024  1.00000  0.95417  0.83876
    ## Pos Pred Value       0.000000      NaN 0.000000      NaN  0.22989  0.25168
    ## Neg Pred Value       0.987021  0.98836 0.975341  0.96315  0.95548  0.92874
    ## Prevalence           0.012928  0.01164 0.024564  0.03685  0.05495  0.10601
    ## Detection Rate       0.000000  0.00000 0.000000  0.00000  0.01293  0.04848
    ## Detection Prevalence 0.003878  0.00000 0.003878  0.00000  0.05624  0.19263
    ## Balanced Accuracy    0.498035  0.50000 0.498012  0.50000  0.59473  0.64804
    ##                      Class: 7 Class: 8 Class: 9 Class: 10 Class: 11 Class: 12
    ## Sensitivity           0.22628  0.15607  0.44318  0.027778   0.62745  0.021505
    ## Specificity           0.92128  0.92940  0.81255  0.991447   0.87059  0.993122
    ## Pos Pred Value        0.21831  0.21774  0.23284  0.250000   0.25498  0.166667
    ## Neg Pred Value        0.92456  0.89740  0.91914  0.908556   0.97068  0.940717
    ## Prevalence            0.08856  0.11183  0.11377  0.093083   0.06593  0.060116
    ## Detection Rate        0.02004  0.01745  0.05042  0.002586   0.04137  0.001293
    ## Detection Prevalence  0.09179  0.08016  0.21655  0.010343   0.16225  0.007757
    ## Balanced Accuracy     0.57378  0.54274  0.62786  0.509612   0.74902  0.507314
    ##                      Class: 13 Class: 14 Class: 15 Class: 16 Class: 17
    ## Sensitivity           0.141304   0.32143  0.122807 0.0163934 0.0370370
    ## Specificity           0.975258   0.91456  0.981879 0.9899058 1.0000000
    ## Pos Pred Value        0.265306   0.17763  0.205882 0.0625000 1.0000000
    ## Neg Pred Value        0.947263   0.95914  0.966953 0.9608099 0.9831824
    ## Prevalence            0.059470   0.05430  0.036846 0.0394312 0.0174531
    ## Detection Rate        0.008403   0.01745  0.004525 0.0006464 0.0006464
    ## Detection Prevalence  0.031674   0.09825  0.021978 0.0103426 0.0006464
    ## Balanced Accuracy     0.558281   0.61799  0.552343 0.5031496 0.5185185
    ##                      Class: 18 Class: 19 Class: 20 Class: 21
    ## Sensitivity           0.000000 0.3333333  0.000000 0.0000000
    ## Specificity           0.990879 0.9987047  1.000000 0.9993532
    ## Pos Pred Value        0.000000 0.3333333       NaN 0.0000000
    ## Neg Pred Value        0.992172 0.9987047  0.998061 0.9993532
    ## Prevalence            0.007757 0.0019392  0.001939 0.0006464
    ## Detection Rate        0.000000 0.0006464  0.000000 0.0000000
    ## Detection Prevalence  0.009050 0.0019392  0.000000 0.0006464
    ## Balanced Accuracy     0.495440 0.6660190  0.500000 0.4996766

The model has a very low accuracy and is not better than random guesses.
