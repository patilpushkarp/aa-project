# Gradient Descent

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
summary(gbm.model)
```

![](reg_gradient_descent_files/figure-gfm/unnamed-chunk-4-1.png)<!-- -->

    ##                                       var     rel.inf
    ## Value                               Value 53.89717110
    ## Lower.Transfer.Value Lower.Transfer.Value 25.25402258
    ## Upper.Transfer.Value Upper.Transfer.Value 15.20755670
    ## Age                                   Age  1.96902586
    ## Mins                                 Mins  1.47108402
    ## Aer.A.90                         Aer.A.90  0.49228855
    ## Distance                         Distance  0.39853344
    ## Apps                                 Apps  0.33109299
    ## Dist.Mins                       Dist.Mins  0.20162203
    ## Weight                             Weight  0.17844083
    ## Gls.xG                             Gls.xG  0.13173928
    ## Ch.C.90                           Ch.C.90  0.09476179
    ## Height                             Height  0.08438834
    ## Tck.R                               Tck.R  0.08332827
    ## Av.Rat                             Av.Rat  0.07740733
    ## Gls.90                             Gls.90  0.07480012
    ## Off                                   Off  0.05273676
    ## Mins.Gm                           Mins.Gm  0.00000000
    ## Gls                                   Gls  0.00000000
    ## Shot..                             Shot..  0.00000000
    ## xG                                     xG  0.00000000
    ## Asts.90                           Asts.90  0.00000000
    ## K.Ps.90                           K.Ps.90  0.00000000
    ## Pas..                               Pas..  0.00000000
    ## Cr.C.A                             Cr.C.A  0.00000000
    ## Drb.90                             Drb.90  0.00000000
    ## Hdr..                               Hdr..  0.00000000
    ## K.Tck                               K.Tck  0.00000000
    ## Fls                                   Fls  0.00000000
    ## PoM                                   PoM  0.00000000

## Model Validation

``` r
# Predict the samples from test data using the model
result <- predict(gbm.model, test)
```

    ## Using 100 trees...

``` r
# Print the RMSE and MAE
cat(paste("RMSE: ", RMSE(result, test$CA), "\n", "MAE: ", MAE(result, test$CA)))
```

    ## RMSE:  8.12950741466064 
    ##  MAE:  6.34950050205738
