# Logistic Regression

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
library(knitr)

# Load helpers
source("./../helpers/helper.R")
```

## Import data

To evaluate the model, there should be a set of which the model has not
seen and for which the labels are known. Hence, it is necessary to split
the data into training and testing set.

``` r
# Read training and testing data
train <- read.csv("./../data/classification_data/intermediates/train.csv")
test <- read.csv("./../data/classification_data/intermediates/test.csv")

# Drop index columns
drops <- c("X.1", "X")
train <- train[, !(names(train) %in% drops)]
test <- test[, !(names(test) %in% drops)]
```

## Model Training

The model is first trained on the training data and then evaluated on
testing data.

``` r
# Model training
multinom.model <- multinom(Rating~., data=train, )
```

    ## # weights:  651 (600 variable)
    ## initial  value 18869.950069 
    ## iter  10 value 17562.557379
    ## iter  20 value 17046.441245
    ## iter  30 value 16888.331863
    ## iter  40 value 16835.134836
    ## iter  50 value 16709.814421
    ## iter  60 value 16398.964065
    ## iter  70 value 15910.131446
    ## iter  80 value 15157.700582
    ## iter  90 value 14305.549703
    ## iter 100 value 13807.595833
    ## final  value 13807.595833 
    ## stopped after 100 iterations

``` r
# summary(multinom.model)
```

## Model Validation

``` r
# Predict the samples from test data using the model
result <- predict(multinom.model, test)

# Print the Confusion matrix
confusion.matrix <- confusionMatrix(as.factor(result), as.factor(test$Rating))
plot.custom.confusion.matrix(confusion.matrix$table)
```

![](class_logistic_regression_files/figure-gfm/unnamed-chunk-5-1.png)<!-- -->

``` r
# Print the accuracy stats of the model
kable(data.frame(confusion.matrix$overall))
```

|                | confusion.matrix.overall |
|:---------------|-------------------------:|
| Accuracy       |                0.2378798 |
| Kappa          |                0.1641406 |
| AccuracyLower  |                0.2168527 |
| AccuracyUpper  |                0.2599064 |
| AccuracyNull   |                0.1215255 |
| AccuracyPValue |                0.0000000 |
| McnemarPValue  |                      NaN |

``` r
# Print validation stats of the model
kable(data.frame(confusion.matrix$byClass))
```

|           | Sensitivity | Specificity | Pos.Pred.Value | Neg.Pred.Value | Precision |    Recall |        F1 | Prevalence | Detection.Rate | Detection.Prevalence | Balanced.Accuracy |
|:----------|------------:|------------:|---------------:|---------------:|----------:|----------:|----------:|-----------:|---------------:|---------------------:|------------------:|
| Class: 1  |   0.0000000 |   0.9960759 |      0.0000000 |      0.9883193 | 0.0000000 | 0.0000000 |       NaN |  0.0116354 |      0.0000000 |            0.0038785 |         0.4980379 |
| Class: 2  |   0.0000000 |   0.9980418 |      0.0000000 |      0.9902850 | 0.0000000 | 0.0000000 |       NaN |  0.0096962 |      0.0000000 |            0.0019392 |         0.4990209 |
| Class: 3  |   0.0000000 |   1.0000000 |            NaN |      0.9728507 |        NA | 0.0000000 |        NA |  0.0271493 |      0.0000000 |            0.0000000 |         0.5000000 |
| Class: 4  |   0.0000000 |   0.9993311 |      0.0000000 |      0.9663648 | 0.0000000 | 0.0000000 |       NaN |  0.0336134 |      0.0000000 |            0.0006464 |         0.4996656 |
| Class: 5  |   0.1794872 |   0.9489449 |      0.1573034 |      0.9561043 | 0.1573034 | 0.1794872 | 0.1676647 |  0.0504202 |      0.0090498 |            0.0575307 |         0.5642160 |
| Class: 6  |   0.4497041 |   0.8011611 |      0.2171429 |      0.9223058 | 0.2171429 | 0.4497041 | 0.2928709 |  0.1092437 |      0.0491273 |            0.2262443 |         0.6254326 |
| Class: 7  |   0.1172414 |   0.9529244 |      0.2048193 |      0.9125683 | 0.2048193 | 0.1172414 | 0.1491228 |  0.0937298 |      0.0109890 |            0.0536522 |         0.5350829 |
| Class: 8  |   0.1987578 |   0.9545455 |      0.3368421 |      0.9111570 | 0.3368421 | 0.1987578 | 0.2500000 |  0.1040724 |      0.0206852 |            0.0614092 |         0.5766516 |
| Class: 9  |   0.3936170 |   0.8226637 |      0.2349206 |      0.9074675 | 0.2349206 | 0.3936170 | 0.2942346 |  0.1215255 |      0.0478345 |            0.2036199 |         0.6081404 |
| Class: 10 |   0.1111111 |   0.9665004 |      0.2539683 |      0.9137466 | 0.2539683 | 0.1111111 | 0.1545894 |  0.0930834 |      0.0103426 |            0.0407240 |         0.5388057 |
| Class: 11 |   0.5181818 |   0.9032707 |      0.2908163 |      0.9607698 | 0.2908163 | 0.5181818 | 0.3725490 |  0.0711054 |      0.0368455 |            0.1266968 |         0.7107263 |
| Class: 12 |   0.2705882 |   0.9432285 |      0.2169811 |      0.9569743 | 0.2169811 | 0.2705882 | 0.2408377 |  0.0549451 |      0.0148675 |            0.0685197 |         0.6069083 |
| Class: 13 |   0.2967033 |   0.9361264 |      0.2250000 |      0.9551507 | 0.2250000 | 0.2967033 | 0.2559242 |  0.0588235 |      0.0174531 |            0.0775695 |         0.6164148 |
| Class: 14 |   0.2272727 |   0.9698424 |      0.3125000 |      0.9541470 | 0.3125000 | 0.2272727 | 0.2631579 |  0.0568843 |      0.0129282 |            0.0413704 |         0.5985575 |
| Class: 15 |   0.1372549 |   0.9926471 |      0.3888889 |      0.9712230 | 0.3888889 | 0.1372549 | 0.2028986 |  0.0329670 |      0.0045249 |            0.0116354 |         0.5649510 |
| Class: 16 |   0.0000000 |   0.9980013 |      0.0000000 |      0.9702073 | 0.0000000 | 0.0000000 |       NaN |  0.0297350 |      0.0000000 |            0.0019392 |         0.4990007 |
| Class: 17 |   0.1000000 |   0.9880557 |      0.1818182 |      0.9763934 | 0.1818182 | 0.1000000 | 0.1290323 |  0.0258565 |      0.0025856 |            0.0142211 |         0.5440279 |
| Class: 18 |   0.0666667 |   0.9947781 |      0.1111111 |      0.9908973 | 0.1111111 | 0.0666667 | 0.0833333 |  0.0096962 |      0.0006464 |            0.0058177 |         0.5307224 |
| Class: 19 |   0.0000000 |   1.0000000 |            NaN |      0.9993536 |        NA | 0.0000000 |        NA |  0.0006464 |      0.0000000 |            0.0000000 |         0.5000000 |
| Class: 20 |   0.0000000 |   1.0000000 |            NaN |      0.9974144 |        NA | 0.0000000 |        NA |  0.0025856 |      0.0000000 |            0.0000000 |         0.5000000 |
| Class: 21 |   0.0000000 |   0.9974076 |      0.0000000 |      0.9974076 | 0.0000000 | 0.0000000 |       NaN |  0.0025856 |      0.0000000 |            0.0025856 |         0.4987038 |

The model has a very low accuracy but is still better than the random
guess. For this case Positive Predictive Value is more important, since
false positives will be highly detrimental for the company and more
correct ratings (positive values) should be identified.

``` r
algorithm <- "Logistic.Regression"
save.class.acc.result(confusion.matrix$overall, algorithm)
save.class.pvv.result(confusion.matrix$byClass, algorithm)
```

## Prediction with Unknown Data

``` r
# Load the data
unk <- read.csv("./../data/classification_data/intermediates/unknown_data.csv")
```

    ## Warning in read.table(file = file, header = header, sep = sep, quote =
    ## quote, : incomplete final line found by readTableHeader on './../data/
    ## classification_data/intermediates/unknown_data.csv'

``` r
dim(unk)
```

    ## [1]  1 29

``` r
# Predict using the built model
prediction <- predict(multinom.model, unk)
prediction
```

    ## [1] 9
    ## Levels: 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21
