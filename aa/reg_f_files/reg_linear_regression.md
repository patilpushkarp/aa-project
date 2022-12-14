# Linear Regression

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

``` r
names(train)
```

    ##  [1] "Apps"                 "Mins"                 "Mins.Gm"             
    ##  [4] "Height"               "Weight"               "Age"                 
    ##  [7] "Av.Rat"               "Gls"                  "Gls.90"              
    ## [10] "Shot.."               "xG"                   "Ch.C.90"             
    ## [13] "Asts.90"              "K.Ps.90"              "Pas.."               
    ## [16] "Cr.C.A"               "Drb.90"               "Distance"            
    ## [19] "Hdr.."                "K.Tck"                "Fls"                 
    ## [22] "PoM"                  "Aer.A.90"             "Off"                 
    ## [25] "Tck.R"                "CA"                   "Gls.xG"              
    ## [28] "Dist.Mins"            "Value"                "Lower.Transfer.Value"
    ## [31] "Upper.Transfer.Value"

## Model Training

The model is first trained on the training data and then evaluated on
testing data.

``` r
# Model training
linear.model <- lm(CA~., data=train)
```

``` r
summary(linear.model)
```

    ## 
    ## Call:
    ## lm(formula = CA ~ ., data = train)
    ## 
    ## Residuals:
    ##     Min      1Q  Median      3Q     Max 
    ## -47.380  -9.358   0.144   9.657  51.063 
    ## 
    ## Coefficients:
    ##                        Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)          -2.860e+01  3.648e+01  -0.784 0.433274    
    ## Apps                 -5.915e-01  1.627e+00  -0.364 0.716305    
    ## Mins                  1.072e-02  1.822e-02   0.588 0.556627    
    ## Mins.Gm              -1.232e-01  2.761e-01  -0.446 0.655530    
    ## Height                9.875e-02  1.217e-01   0.811 0.417526    
    ## Weight               -2.090e-02  1.044e-01  -0.200 0.841413    
    ## Age                   3.010e-01  1.434e-01   2.098 0.036169 *  
    ## Av.Rat                1.271e+01  2.804e+00   4.533 6.63e-06 ***
    ## Gls                   1.242e+00  1.485e+00   0.836 0.403177    
    ## Gls.90               -3.949e+01  2.056e+01  -1.921 0.055070 .  
    ## Shot..                5.794e+00  3.003e+00   1.929 0.053997 .  
    ## xG                   -1.027e+00  7.470e-01  -1.375 0.169619    
    ## Ch.C.90               8.856e-01  1.271e+00   0.697 0.486023    
    ## Asts.90              -8.136e+00  5.714e+00  -1.424 0.154817    
    ## K.Ps.90              -9.059e+00  1.729e+00  -5.240 2.02e-07 ***
    ## Pas..                 3.700e+01  7.537e+00   4.909 1.09e-06 ***
    ## Cr.C.A               -2.865e+00  2.069e+00  -1.385 0.166539    
    ## Drb.90                1.827e+00  1.327e+00   1.376 0.169120    
    ## Distance              5.452e-02  2.781e-02   1.961 0.050249 .  
    ## Hdr..                -7.517e+00  4.093e+00  -1.836 0.066640 .  
    ## K.Tck                 5.145e-01  3.948e-01   1.303 0.192845    
    ## Fls                  -1.171e-01  7.330e-02  -1.597 0.110616    
    ## PoM                  -2.639e-01  5.786e-01  -0.456 0.648481    
    ## Aer.A.90             -1.175e+00  1.827e-01  -6.433 2.06e-10 ***
    ## Off                  -3.978e-01  4.007e-01  -0.993 0.321144    
    ## Tck.R                -2.626e+00  2.697e+00  -0.974 0.330515    
    ## Gls.xG                1.043e-01  8.843e-02   1.180 0.238374    
    ## Dist.Mins             1.610e+02  4.827e+01   3.336 0.000886 ***
    ## Value                 5.186e-06  3.011e-06   1.723 0.085312 .  
    ## Lower.Transfer.Value -3.551e-06  1.517e-06  -2.341 0.019452 *  
    ## Upper.Transfer.Value -1.566e-06  1.501e-06  -1.043 0.297039    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 13.31 on 867 degrees of freedom
    ## Multiple R-squared:  0.589,  Adjusted R-squared:  0.5748 
    ## F-statistic: 41.41 on 30 and 867 DF,  p-value: < 2.2e-16

``` r
# Model training with new significant variable
new.train <- train[, c(6, 7, 14, 15, 23, 26, 28, 30)]
new.linear.model <- lm(CA~., data=new.train)
summary(new.linear.model)
```

    ## 
    ## Call:
    ## lm(formula = CA ~ ., data = new.train)
    ## 
    ## Residuals:
    ##      Min       1Q   Median       3Q      Max 
    ## -102.019  -11.207    1.741   11.058   49.145 
    ## 
    ## Coefficients:
    ##                        Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)          -1.373e+01  1.904e+01  -0.721    0.471    
    ## Age                   1.649e-01  1.601e-01   1.030    0.303    
    ## Av.Rat                1.368e+01  2.640e+00   5.182 2.71e-07 ***
    ## K.Ps.90              -1.318e+01  1.596e+00  -8.255 5.45e-16 ***
    ## Pas..                 2.485e+01  5.730e+00   4.336 1.61e-05 ***
    ## Aer.A.90             -1.365e+00  1.667e-01  -8.185 9.39e-16 ***
    ## Dist.Mins             2.298e+02  2.817e+01   8.155 1.18e-15 ***
    ## Lower.Transfer.Value  3.845e-07  2.033e-08  18.916  < 2e-16 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 15.28 on 890 degrees of freedom
    ## Multiple R-squared:  0.4442, Adjusted R-squared:  0.4398 
    ## F-statistic: 101.6 on 7 and 890 DF,  p-value: < 2.2e-16

## Model Validation

``` r
# Predict the samples from test data using the model
result <- predict(linear.model, test)

# Print the RMSE and MAE
cat(paste("RMSE: ", RMSE(result, test$CA), "\n", "MSE: ", RMSE(result, test$CA)^2, "\n", "MAE: ", MAE(result, test$CA)))
```

    ## RMSE:  12.6975984315159 
    ##  MSE:  161.229005928036 
    ##  MAE:  10.4290241532838

``` r
# Save the results
save.reg.result(RMSE(result, test$CA), MAE(result, test$CA), "Linear Regression")
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
prediction <- predict(linear.model, unk)
prediction
```

    ##        1 
    ## 100.8627
