# Random Forest

Since the data has been cleaned, it can now be used to create the
models.

``` r
# Load libraries
library(caret)
```

    ## Loading required package: ggplot2

    ## Loading required package: lattice

``` r
library(randomForest)
```

    ## randomForest 4.7-1.1

    ## Type rfNews() to see new features/changes/bug fixes.

    ## 
    ## Attaching package: 'randomForest'

    ## The following object is masked from 'package:ggplot2':
    ## 
    ##     margin

``` r
library(MASS)

# Load helpers
source("./../helpers/helper.R")
```

## Import data

To evaluate the model, there should be a set of which the model has not
seen and for which the labels are known. Hence, it is necessary to split
the data into training and testing set.

``` r
# Read training and testing data
train <- read.csv("./../data/regression_data/intermediates/train.csv")
test <- read.csv("./../data/regression_data/intermediates/test.csv")
```

## Model Training

The model is first trained on the training data and then evaluated on
testing data.

``` r
# Model training
rf.model <- randomForest(CA~., data=train)
```

``` r
summary(rf.model)
```

    ##                 Length Class  Mode     
    ## call              3    -none- call     
    ## type              1    -none- character
    ## predicted       898    -none- numeric  
    ## mse             500    -none- numeric  
    ## rsq             500    -none- numeric  
    ## oob.times       898    -none- numeric  
    ## importance       30    -none- numeric  
    ## importanceSD      0    -none- NULL     
    ## localImportance   0    -none- NULL     
    ## proximity         0    -none- NULL     
    ## ntree             1    -none- numeric  
    ## mtry              1    -none- numeric  
    ## forest           11    -none- list     
    ## coefs             0    -none- NULL     
    ## y               898    -none- numeric  
    ## test              0    -none- NULL     
    ## inbag             0    -none- NULL     
    ## terms             3    terms  call

## Model Validation

``` r
# Predict the samples from test data using the model
result <- predict(rf.model, test)

# Print the RMSE and MAE
cat(paste("RMSE: ", RMSE(result, test$CA), "\n", "MAE: ", MAE(result, test$CA)))
```

    ## RMSE:  8.06816272785225 
    ##  MAE:  6.1587538116592

``` r
varImp(rf.model)
```

    ##                          Overall
    ## Apps                   5561.0911
    ## Mins                   6380.8248
    ## Mins.Gm                2017.4140
    ## Height                 1992.8612
    ## Weight                 2088.4085
    ## Age                    4481.3923
    ## Av.Rat                 7933.9759
    ## Gls                     376.1644
    ## Gls.90                  662.7810
    ## Shot..                 1073.9416
    ## xG                     1536.6192
    ## Ch.C.90                1385.5363
    ## Asts.90                 942.0452
    ## K.Ps.90                2627.1438
    ## Pas..                  3258.7395
    ## Cr.C.A                  986.3250
    ## Drb.90                 1183.2440
    ## Distance               7271.4434
    ## Hdr..                  1607.2684
    ## K.Tck                   699.6000
    ## Fls                    1871.0538
    ## PoM                     976.0439
    ## Aer.A.90               5132.4956
    ## Off                     609.7598
    ## Tck.R                  1966.7884
    ## Gls.xG                  936.4560
    ## Dist.Mins              4839.9684
    ## Value                130993.0707
    ## Lower.Transfer.Value  80992.8334
    ## Upper.Transfer.Value  87949.5836

``` r
# Save the results
save.reg.result(RMSE(result, test$CA), MAE(result, test$CA), "Random Forest Regression")
```
