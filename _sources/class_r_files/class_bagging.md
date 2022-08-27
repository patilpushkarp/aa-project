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

# Drop index columns
drops <- c("X.1", "X")
train <- train[, !(names(train) %in% drops)]
test <- test[, !(names(test) %in% drops)]
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
| Accuracy       |                0.5824176 |
| Kappa          |                0.5469846 |
| AccuracyLower  |                0.5573823 |
| AccuracyUpper  |                0.6071394 |
| AccuracyNull   |                0.1215255 |
| AccuracyPValue |                0.0000000 |
| McnemarPValue  |                      NaN |

``` r
# Print validation stats of the model
kable(data.frame(confusion.matrix$byClass))
```

|           | Sensitivity | Specificity | Pos.Pred.Value | Neg.Pred.Value | Precision |    Recall |        F1 | Prevalence | Detection.Rate | Detection.Prevalence | Balanced.Accuracy |
|:----------|------------:|------------:|---------------:|---------------:|----------:|----------:|----------:|-----------:|---------------:|---------------------:|------------------:|
| Class: 1  |   0.6111111 |   0.9947678 |      0.5789474 |      0.9954188 | 0.5789474 | 0.6111111 | 0.5945946 |  0.0116354 |      0.0071105 |            0.0122818 |         0.8029395 |
| Class: 2  |   0.7333333 |   0.9993473 |      0.9166667 |      0.9973941 | 0.9166667 | 0.7333333 | 0.8148148 |  0.0096962 |      0.0071105 |            0.0077569 |         0.8663403 |
| Class: 3  |   0.5714286 |   0.9900332 |      0.6153846 |      0.9880637 | 0.6153846 | 0.5714286 | 0.5925926 |  0.0271493 |      0.0155139 |            0.0252101 |         0.7807309 |
| Class: 4  |   0.6346154 |   0.9899666 |      0.6875000 |      0.9873249 | 0.6875000 | 0.6346154 | 0.6600000 |  0.0336134 |      0.0213316 |            0.0310278 |         0.8122910 |
| Class: 5  |   0.6410256 |   0.9782165 |      0.6097561 |      0.9808874 | 0.6097561 | 0.6410256 | 0.6250000 |  0.0504202 |      0.0323206 |            0.0530058 |         0.8096211 |
| Class: 6  |   0.6449704 |   0.9470247 |      0.5989011 |      0.9560440 | 0.5989011 | 0.6449704 | 0.6210826 |  0.1092437 |      0.0704590 |            0.1176471 |         0.7959975 |
| Class: 7  |   0.6206897 |   0.9664765 |      0.6569343 |      0.9609929 | 0.6569343 | 0.6206897 | 0.6382979 |  0.0937298 |      0.0581771 |            0.0885585 |         0.7935831 |
| Class: 8  |   0.5652174 |   0.9458874 |      0.5481928 |      0.9493121 | 0.5481928 | 0.5652174 | 0.5565749 |  0.1040724 |      0.0588235 |            0.1073045 |         0.7555524 |
| Class: 9  |   0.5744681 |   0.9352465 |      0.5510204 |      0.9407846 | 0.5510204 | 0.5744681 | 0.5625000 |  0.1215255 |      0.0698125 |            0.1266968 |         0.7548573 |
| Class: 10 |   0.6180556 |   0.9700641 |      0.6793893 |      0.9611582 | 0.6793893 | 0.6180556 | 0.6472727 |  0.0930834 |      0.0575307 |            0.0846800 |         0.7940599 |
| Class: 11 |   0.7545455 |   0.9728601 |      0.6803279 |      0.9810526 | 0.6803279 | 0.7545455 | 0.7155172 |  0.0711054 |      0.0536522 |            0.0788623 |         0.8637028 |
| Class: 12 |   0.5647059 |   0.9623803 |      0.4660194 |      0.9743767 | 0.4660194 | 0.5647059 | 0.5106383 |  0.0549451 |      0.0310278 |            0.0665805 |         0.7635431 |
| Class: 13 |   0.4945055 |   0.9697802 |      0.5056180 |      0.9684499 | 0.5056180 | 0.4945055 | 0.5000000 |  0.0588235 |      0.0290886 |            0.0575307 |         0.7321429 |
| Class: 14 |   0.5000000 |   0.9787526 |      0.5866667 |      0.9701087 | 0.5866667 | 0.5000000 | 0.5398773 |  0.0568843 |      0.0284421 |            0.0484809 |         0.7393763 |
| Class: 15 |   0.3529412 |   0.9806150 |      0.3829787 |      0.9780000 | 0.3829787 | 0.3529412 | 0.3673469 |  0.0329670 |      0.0116354 |            0.0303814 |         0.6667781 |
| Class: 16 |   0.5217391 |   0.9773484 |      0.4137931 |      0.9852250 | 0.4137931 | 0.5217391 | 0.4615385 |  0.0297350 |      0.0155139 |            0.0374919 |         0.7495438 |
| Class: 17 |   0.3250000 |   0.9933643 |      0.5652174 |      0.9822835 | 0.5652174 | 0.3250000 | 0.4126984 |  0.0258565 |      0.0084034 |            0.0148675 |         0.6591821 |
| Class: 18 |   0.5333333 |   0.9967363 |      0.6153846 |      0.9954368 | 0.6153846 | 0.5333333 | 0.5714286 |  0.0096962 |      0.0051713 |            0.0084034 |         0.7650348 |
| Class: 19 |   0.0000000 |   1.0000000 |            NaN |      0.9993536 |        NA | 0.0000000 |        NA |  0.0006464 |      0.0000000 |            0.0000000 |         0.5000000 |
| Class: 20 |   0.0000000 |   0.9980557 |      0.0000000 |      0.9974093 | 0.0000000 | 0.0000000 |       NaN |  0.0025856 |      0.0000000 |            0.0019392 |         0.4990279 |
| Class: 21 |   0.5000000 |   1.0000000 |      1.0000000 |      0.9987055 | 1.0000000 | 0.5000000 | 0.6666667 |  0.0025856 |      0.0012928 |            0.0012928 |         0.7500000 |

``` r
# Get the feature importance
kable(varImp(bag.model))
```

|                                                |    Overall |
|:-----------------------------------------------|-----------:|
| Asset.Turnover                                 | 1403.89823 |
| Binary.Rating                                  |  404.07602 |
| Current.Ratio                                  | 1728.49544 |
| Debt.Equity.Ratio                              | 1627.74969 |
| Free.Cash.Flow.Per.Share                       |  997.97976 |
| Gross.Margin                                   | 1587.06878 |
| Long.term.Debt…Capital                         | 1730.22150 |
| Net.Profit.Margin                              | 1552.14383 |
| Operating.Cash.Flow.Per.Share                  | 1046.92139 |
| Rating.Agency_Egan.Jones.Ratings.Company       |  243.46814 |
| Rating.Agency_Fitch.Ratings                    |  174.95461 |
| Rating.Agency_Moody.s.Investors.Service        |  341.32343 |
| Rating.Agency_Standard…Poor.s.Ratings.Services |  392.18583 |
| Return.On.Tangible.Equity                      | 1268.11999 |
| ROA…Return.On.Assets                           | 1272.29294 |
| ROE…Return.On.Equity                           | 1388.80697 |
| ROI…Return.On.Investment                       | 1197.04377 |
| Sector_BusEq                                   |  194.94247 |
| Sector_Chems                                   |   94.33077 |
| Sector_Durbl                                   |   67.48308 |
| Sector_Enrgy                                   |  159.51631 |
| Sector_Hlth                                    |  202.51301 |
| Sector_Manuf                                   |  184.68844 |
| Sector_Money                                   |   89.04967 |
| Sector_NoDur                                   |  115.01825 |
| Sector_Other                                   |  247.68894 |
| Sector_Shops                                   |  182.02520 |
| Sector_Telcm                                   |   96.52557 |
| Sector_Utils                                   |  114.58410 |

``` r
# Save the results
algorithm <- "Classification.with.Bagging"
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
prediction <- predict(bag.model, unk)
prediction
```

    ## [1] 9
    ## Levels: 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21
