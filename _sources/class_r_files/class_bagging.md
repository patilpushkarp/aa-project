# Classification with Bagging

``` r
# Load libraries
library(ipred)
library(caret)
```

    ## Loading required package: ggplot2

    ## Loading required package: lattice

``` r
library(knitr)

# Load helpers
source("./../helpers/helper.R")
```

## Import Data

``` r
# Read training and testing data
train <- read.csv("./../data/classification_data/intermediates/train.csv")
test <- read.csv("./../data/classification_data/intermediates/test.csv")
```

## Model Training

``` r
# Model training
bag.model = bagging(as.factor(Rating)~., data=train)
bag.model
```

    ## 
    ## Bagging classification trees with 25 bootstrap replications 
    ## 
    ## Call: bagging.data.frame(formula = as.factor(Rating) ~ ., data = train)

## Model Validation

``` r
# Predict the samples from test data using the model
result <- predict(bag.model, test, type="class")

# Print the Confusion matrix
confusion.matrix <- confusionMatrix(as.factor(result), as.factor(test$Rating))
plot.custom.confusion.matrix(confusion.matrix$table)
```

![](class_bagging_files/figure-gfm/unnamed-chunk-4-1.png)<!-- -->

``` r
# Print the accuracy stats of the model
kable(data.frame(confusion.matrix$overall))
```

|                | confusion.matrix.overall |
|:---------------|-------------------------:|
| Accuracy       |                0.5720750 |
| Kappa          |                0.5354938 |
| AccuracyLower  |                0.5469789 |
| AccuracyUpper  |                0.5968969 |
| AccuracyNull   |                0.1215255 |
| AccuracyPValue |                0.0000000 |
| McnemarPValue  |                      NaN |

``` r
# Print validation stats of the model
kable(data.frame(confusion.matrix$byClass))
```

|           | Sensitivity | Specificity | Pos.Pred.Value | Neg.Pred.Value | Precision |    Recall |        F1 | Prevalence | Detection.Rate | Detection.Prevalence | Balanced.Accuracy |
|:----------|------------:|------------:|---------------:|---------------:|----------:|----------:|----------:|-----------:|---------------:|---------------------:|------------------:|
| Class: 1  |   0.5000000 |   0.9960759 |      0.6000000 |      0.9941253 | 0.6000000 | 0.5000000 | 0.5454545 |  0.0116354 |      0.0058177 |            0.0096962 |         0.7480379 |
| Class: 2  |   0.7333333 |   0.9993473 |      0.9166667 |      0.9973941 | 0.9166667 | 0.7333333 | 0.8148148 |  0.0096962 |      0.0071105 |            0.0077569 |         0.8663403 |
| Class: 3  |   0.5952381 |   0.9913621 |      0.6578947 |      0.9887343 | 0.6578947 | 0.5952381 | 0.6250000 |  0.0271493 |      0.0161603 |            0.0245637 |         0.7933001 |
| Class: 4  |   0.6538462 |   0.9953177 |      0.8292683 |      0.9880478 | 0.8292683 | 0.6538462 | 0.7311828 |  0.0336134 |      0.0219780 |            0.0265029 |         0.8245819 |
| Class: 5  |   0.6666667 |   0.9761743 |      0.5977011 |      0.9821918 | 0.5977011 | 0.6666667 | 0.6303030 |  0.0504202 |      0.0336134 |            0.0562379 |         0.8214205 |
| Class: 6  |   0.6390533 |   0.9521045 |      0.6206897 |      0.9555717 | 0.6206897 | 0.6390533 | 0.6297376 |  0.1092437 |      0.0698125 |            0.1124758 |         0.7955789 |
| Class: 7  |   0.6344828 |   0.9650499 |      0.6524823 |      0.9623044 | 0.6524823 | 0.6344828 | 0.6433566 |  0.0937298 |      0.0594699 |            0.0911441 |         0.7997663 |
| Class: 8  |   0.6024845 |   0.9430014 |      0.5511364 |      0.9533187 | 0.5511364 | 0.6024845 | 0.5756677 |  0.1040724 |      0.0627020 |            0.1137686 |         0.7727430 |
| Class: 9  |   0.5904255 |   0.9308315 |      0.5414634 |      0.9426230 | 0.5414634 | 0.5904255 | 0.5648855 |  0.1215255 |      0.0717518 |            0.1325145 |         0.7606285 |
| Class: 10 |   0.5833333 |   0.9722024 |      0.6829268 |      0.9578652 | 0.6829268 | 0.5833333 | 0.6292135 |  0.0930834 |      0.0542986 |            0.0795087 |         0.7777679 |
| Class: 11 |   0.6636364 |   0.9707724 |      0.6347826 |      0.9741620 | 0.6347826 | 0.6636364 | 0.6488889 |  0.0711054 |      0.0471881 |            0.0743374 |         0.8172044 |
| Class: 12 |   0.5411765 |   0.9541724 |      0.4070796 |      0.9728033 | 0.4070796 | 0.5411765 | 0.4646465 |  0.0549451 |      0.0297350 |            0.0730446 |         0.7476744 |
| Class: 13 |   0.4505495 |   0.9711538 |      0.4939759 |      0.9658470 | 0.4939759 | 0.4505495 | 0.4712644 |  0.0588235 |      0.0265029 |            0.0536522 |         0.7108516 |
| Class: 14 |   0.4545455 |   0.9705278 |      0.4819277 |      0.9672131 | 0.4819277 | 0.4545455 | 0.4678363 |  0.0568843 |      0.0258565 |            0.0536522 |         0.7125366 |
| Class: 15 |   0.4117647 |   0.9799465 |      0.4117647 |      0.9799465 | 0.4117647 | 0.4117647 | 0.4117647 |  0.0329670 |      0.0135747 |            0.0329670 |         0.6958556 |
| Class: 16 |   0.3260870 |   0.9806795 |      0.3409091 |      0.9793746 | 0.3409091 | 0.3260870 | 0.3333333 |  0.0297350 |      0.0096962 |            0.0284421 |         0.6533833 |
| Class: 17 |   0.4000000 |   0.9920372 |      0.5714286 |      0.9842001 | 0.5714286 | 0.4000000 | 0.4705882 |  0.0258565 |      0.0103426 |            0.0180995 |         0.6960186 |
| Class: 18 |   0.5333333 |   0.9967363 |      0.6153846 |      0.9954368 | 0.6153846 | 0.5333333 | 0.5714286 |  0.0096962 |      0.0051713 |            0.0084034 |         0.7650348 |
| Class: 19 |   0.0000000 |   0.9993532 |      0.0000000 |      0.9993532 | 0.0000000 | 0.0000000 |       NaN |  0.0006464 |      0.0000000 |            0.0006464 |         0.4996766 |
| Class: 20 |   0.0000000 |   0.9987038 |      0.0000000 |      0.9974110 | 0.0000000 | 0.0000000 |       NaN |  0.0025856 |      0.0000000 |            0.0012928 |         0.4993519 |
| Class: 21 |   0.5000000 |   1.0000000 |      1.0000000 |      0.9987055 | 1.0000000 | 0.5000000 | 0.6666667 |  0.0025856 |      0.0012928 |            0.0012928 |         0.7500000 |

``` r
# Get the feature importance
kable(varImp(bag.model))
```

|                                                |    Overall |
|:-----------------------------------------------|-----------:|
| Asset.Turnover                                 | 1262.30505 |
| Binary.Rating                                  |  401.47666 |
| Current.Ratio                                  | 1662.39423 |
| Debt.Equity.Ratio                              | 1418.60437 |
| Free.Cash.Flow.Per.Share                       |  883.82731 |
| Gross.Margin                                   | 1452.36683 |
| Long.term.Debt…Capital                         | 1579.15390 |
| Net.Profit.Margin                              | 1411.48242 |
| Operating.Cash.Flow.Per.Share                  |  921.92747 |
| Rating.Agency_Egan.Jones.Ratings.Company       |  232.24354 |
| Rating.Agency_Fitch.Ratings                    |  173.98798 |
| Rating.Agency_Moody.s.Investors.Service        |  341.02880 |
| Rating.Agency_Standard…Poor.s.Ratings.Services |  386.84704 |
| Return.On.Tangible.Equity                      | 1120.97422 |
| ROA…Return.On.Assets                           | 1157.03232 |
| ROE…Return.On.Equity                           | 1204.09562 |
| ROI…Return.On.Investment                       | 1085.13034 |
| Sector_BusEq                                   |  207.80436 |
| Sector_Chems                                   |   93.06316 |
| Sector_Durbl                                   |   81.19705 |
| Sector_Enrgy                                   |  158.80023 |
| Sector_Hlth                                    |  191.29762 |
| Sector_Manuf                                   |  186.13030 |
| Sector_Money                                   |   66.49602 |
| Sector_NoDur                                   |  119.47753 |
| Sector_Other                                   |  222.88926 |
| Sector_Shops                                   |  180.05233 |
| Sector_Telcm                                   |   82.71290 |
| Sector_Utils                                   |  106.24327 |
| X                                              | 2065.63541 |
| X.1                                            | 2253.02835 |

``` r
# Save the results
algorithm <- "Classification.with.Bagging"
save.class.acc.result(confusion.matrix$overall, algorithm)
save.class.pvv.result(confusion.matrix$byClass, algorithm)
```
