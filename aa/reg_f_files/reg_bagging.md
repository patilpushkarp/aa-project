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

    ## RMSE:  9.22106263617327 
    ##  MAE:  7.18860072900238

``` r
varImp(bag.model)
```

    ##                          Overall
    ## Aer.A.90             0.088468230
    ## Age                  0.056721189
    ## Apps                 0.284581590
    ## Asts.90              0.003167350
    ## Av.Rat               0.507108418
    ## Ch.C.90              0.009960341
    ## Cr.C.A               0.000000000
    ## Dist.Mins            0.111920828
    ## Distance             0.215753224
    ## Drb.90               0.000000000
    ## Fls                  0.008273462
    ## Gls                  0.000000000
    ## Gls.90               0.000000000
    ## Gls.xG               0.000000000
    ## Hdr..                0.014897457
    ## Height               0.008604565
    ## K.Ps.90              0.048569780
    ## K.Tck                0.017051828
    ## Lower.Transfer.Value 2.258780958
    ## Mins                 0.395633223
    ## Mins.Gm              0.005865406
    ## Off                  0.000000000
    ## Pas..                0.019970066
    ## PoM                  0.014902381
    ## Shot..               0.000000000
    ## Tck.R                0.031488562
    ## Upper.Transfer.Value 2.338556295
    ## Value                2.452467445
    ## Weight               0.014293134
    ## xG                   0.028432393

``` r
# Save the results
save.reg.result(RMSE(result, test$CA), MAE(result, test$CA), "Regression with Bagging")
```
