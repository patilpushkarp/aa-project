# Gradient Boosting

Since the data has been cleaned, it can now be used to create the
models.

``` r
# Load libraries
library(caret)
```

    ## Loading required package: ggplot2

    ## Loading required package: lattice

``` r
library(gbm)
```

    ## Loaded gbm 2.1.8

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
gbm.model <- gbm(CA~., data=train)
```

    ## Distribution not specified, assuming gaussian ...

``` r
gbm.model
```

    ## gbm(formula = CA ~ ., data = train)
    ## A gradient boosted model with gaussian loss function.
    ## 100 iterations were performed.
    ## There were 30 predictors of which 18 had non-zero influence.

``` r
summary(gbm.model)
```

![](reg_gradient_descent_files/figure-gfm/unnamed-chunk-4-1.png)<!-- -->

    ##                                       var     rel.inf
    ## Value                               Value 65.54059290
    ## Lower.Transfer.Value Lower.Transfer.Value 18.66319064
    ## Upper.Transfer.Value Upper.Transfer.Value  9.89414279
    ## Age                                   Age  2.03751841
    ## Mins                                 Mins  1.36118540
    ## Distance                         Distance  0.53257436
    ## Aer.A.90                         Aer.A.90  0.47911331
    ## Apps                                 Apps  0.45873175
    ## Off                                   Off  0.17072448
    ## Gls.xG                             Gls.xG  0.14412651
    ## Height                             Height  0.14092344
    ## Av.Rat                             Av.Rat  0.10941144
    ## Weight                             Weight  0.10911263
    ## Tck.R                               Tck.R  0.08159985
    ## Gls.90                             Gls.90  0.07722888
    ## Shot..                             Shot..  0.07268679
    ## Ch.C.90                           Ch.C.90  0.06977550
    ## K.Ps.90                           K.Ps.90  0.05736093
    ## Mins.Gm                           Mins.Gm  0.00000000
    ## Gls                                   Gls  0.00000000
    ## xG                                     xG  0.00000000
    ## Asts.90                           Asts.90  0.00000000
    ## Pas..                               Pas..  0.00000000
    ## Cr.C.A                             Cr.C.A  0.00000000
    ## Drb.90                             Drb.90  0.00000000
    ## Hdr..                               Hdr..  0.00000000
    ## K.Tck                               K.Tck  0.00000000
    ## Fls                                   Fls  0.00000000
    ## PoM                                   PoM  0.00000000
    ## Dist.Mins                       Dist.Mins  0.00000000

## Model Validation

``` r
# Predict the samples from test data using the model
result <- predict(gbm.model, test)
```

    ## Using 100 trees...

``` r
# Print the RMSE and MAE
cat(paste("RMSE: ", RMSE(result, test$CA), "\n", "MSE: ", RMSE(result, test$CA)^2, "\n", "MAE: ", MAE(result, test$CA)))
```

    ## RMSE:  8.01241834544305 
    ##  MSE:  64.1988477423924 
    ##  MAE:  6.26234884249979

``` r
# Save the results
save.reg.result(RMSE(result, test$CA), MAE(result, test$CA), "Gradient Descent Regression")
```

## Prediction with Unknown Data

``` r
# Load the data
unk <- read.csv("./../data/regression_data/intermediates/unknown_data.csv")
```

    ## Warning in read.table(file = file, header = header, sep = sep, quote = quote, :
    ## incomplete final line found by readTableHeader on './../data/regression_data/
    ## intermediates/unknown_data.csv'

``` r
dim(unk)
```

    ## [1]  1 30

``` r
# Predict using the built model
prediction <- predict(gbm.model, unk)
```

    ## Using 100 trees...

``` r
prediction
```

    ## [1] 99.08949
