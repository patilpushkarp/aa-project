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

    ## RMSE:  8.05180116776778 
    ##  MAE:  6.11829372197309

``` r
varImp(rf.model)
```

    ##                          Overall
    ## Apps                   5308.9349
    ## Mins                   6674.7274
    ## Mins.Gm                1995.7667
    ## Height                 1961.3602
    ## Weight                 2097.4232
    ## Age                    4662.6019
    ## Av.Rat                 8049.7986
    ## Gls                     337.3487
    ## Gls.90                  764.9776
    ## Shot..                 1085.8593
    ## xG                     1542.2896
    ## Ch.C.90                1545.1869
    ## Asts.90                 935.1214
    ## K.Ps.90                2525.1310
    ## Pas..                  3173.3485
    ## Cr.C.A                  959.9266
    ## Drb.90                 1127.4292
    ## Distance               7375.4053
    ## Hdr..                  1447.8460
    ## K.Tck                   662.7340
    ## Fls                    1853.7221
    ## PoM                     824.2982
    ## Aer.A.90               4868.0758
    ## Off                     578.6933
    ## Tck.R                  1766.3690
    ## Gls.xG                  826.4383
    ## Dist.Mins              4444.6905
    ## Value                116502.3158
    ## Lower.Transfer.Value  82531.6406
    ## Upper.Transfer.Value 102640.6963
