# Bagging

Since the data has been cleaned, it can now be used to create the
models.

``` r
# Load libraries
library(caret)
```

    ## Loading required package: ggplot2

    ## Loading required package: lattice

``` r
library(ipred)
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
bag.model <- bagging(CA~., data=train)
```

``` r
summary(bag.model)
```

    ##        Length Class      Mode   
    ## y      898    -none-     numeric
    ## X       30    data.frame list   
    ## mtrees  25    -none-     list   
    ## OOB      1    -none-     logical
    ## comb     1    -none-     logical
    ## call     3    -none-     call

## Model Validation

``` r
# Predict the samples from test data using the model
result <- predict(bag.model, test)

# Print the RMSE and MAE
cat(paste("RMSE: ", RMSE(result, test$CA), "\n", "MAE: ", MAE(result, test$CA)))
```

    ## RMSE:  9.12068202580554 
    ##  MAE:  7.04255118678443

``` r
varImp(bag.model)
```

    ##                          Overall
    ## Aer.A.90             0.107736943
    ## Age                  0.059023129
    ## Apps                 0.302153375
    ## Asts.90              0.000000000
    ## Av.Rat               0.497320831
    ## Ch.C.90              0.003620830
    ## Cr.C.A               0.000000000
    ## Dist.Mins            0.054128097
    ## Distance             0.183935379
    ## Drb.90               0.000000000
    ## Fls                  0.011603811
    ## Gls                  0.000000000
    ## Gls.90               0.000000000
    ## Gls.xG               0.000000000
    ## Hdr..                0.018199061
    ## Height               0.000000000
    ## K.Ps.90              0.016163552
    ## K.Tck                0.000000000
    ## Lower.Transfer.Value 2.174799288
    ## Mins                 0.366249006
    ## Mins.Gm              0.011585111
    ## Off                  0.003825671
    ## Pas..                0.013879425
    ## PoM                  0.010037013
    ## Shot..               0.006076557
    ## Tck.R                0.007617579
    ## Upper.Transfer.Value 2.264646293
    ## Value                2.343813165
    ## Weight               0.022293505
    ## xG                   0.019860573
