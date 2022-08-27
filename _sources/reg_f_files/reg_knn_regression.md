# kNN Regression

Since the data has been cleaned, it can now be used to create the
models.

``` r
# Load libraries
library(caret)
```

    ## Loading required package: ggplot2

    ## Loading required package: lattice

``` r
library(e1071)
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
?knnreg
knn.model <- knnreg(CA~., data=train)
```

``` r
summary(knn.model)
```

    ##         Length Class  Mode   
    ## learn   2      -none- list   
    ## k       1      -none- numeric
    ## terms   3      terms  call   
    ## xlevels 0      -none- list   
    ## theDots 0      -none- list

## Model Validation

``` r
# Predict the samples from test data using the model
result <- predict(knn.model, test)

# Print the RMSE and MAE
cat(paste("RMSE: ", RMSE(result, test$CA), "\n", "MSE: ", RMSE(result, test$CA)^2, "\n", "MAE: ", MAE(result, test$CA)))
```

    ## RMSE:  9.85169422012323 
    ##  MSE:  97.0558790068095 
    ##  MAE:  7.55440600754502

``` r
# Save the results
save.reg.result(RMSE(result, test$CA), MAE(result, test$CA), "kNN Regression")
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
prediction <- predict(knn.model, unk)
prediction
```

    ## [1] 96.2
