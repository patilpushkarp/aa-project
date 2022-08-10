# Bagging

``` r
# Load libraries
library(ipred)
library(caret)
```

    ## Loading required package: ggplot2

    ## Loading required package: lattice

## Import Data

``` r
# Read training and testing data
train <- read.csv("./../data/classification_data/intermediates/train.csv")
test <- read.csv("./../data/classification_data/intermediates/test.csv")
```

## Model Training

``` r
bag.model = bagging(as.factor(Rating)~., data=train)
bag.model
```

    ## 
    ## Bagging classification trees with 25 bootstrap replications 
    ## 
    ## Call: bagging.data.frame(formula = as.factor(Rating) ~ ., data = train)

## Model Validation

``` r
result <- predict(bag.model, test, type="class")

confusionMatrix(as.factor(result), as.factor(test$Rating))
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17
    ##         1    8   0   0   1   1   0   0   0   0   0   0   0   0   0   0   0   0
    ##         2    3  16   1   1   1   1   0   1   0   0   0   0   0   0   0   0   0
    ##         3    1   0  20   3   4   5   2   0   0   1   0   0   0   0   0   0   0
    ##         4    1   0   1  36  10   2   1   0   0   0   0   0   0   0   0   0   0
    ##         5    1   1   1   4  66  12   3   1   1   1   0   0   0   0   0   0   0
    ##         6    1   0   6   1   3 113  13  13  10   5   0   0   0   0   0   0   0
    ##         7    0   0   1   2   3  14  90  22   8   3   0   0   0   0   0   0   0
    ##         8    0   0   0   0   3  15  17 110  33  10   0   0   0   0   0   0   0
    ##         9    1   0   1   0   3  10   4  29  96  34   0   0   0   0   0   0   0
    ##         10   0   0   0   0   5   3   3   3  22  89   0   0   0   0   0   0   0
    ##         11   0   0   0   0   0   0   0   0   0   0  71  20   7   5   2   2   0
    ##         12   0   0   0   0   0   0   0   0   0   0  26  46  14  12   3   4   1
    ##         13   0   0   0   0   0   0   0   0   0   0   6  11  33  10   9   2   3
    ##         14   0   0   0   0   0   0   0   0   0   0   1   5  16  48  16   9   5
    ##         15   0   0   0   0   0   0   0   0   0   0   3   0   0  10  25   5   3
    ##         16   0   0   0   0   0   0   0   0   0   0   3   2   7   6   6  28   2
    ##         17   0   0   0   0   0   0   0   0   0   0   0   1   2   1   3   6  14
    ##         18   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   2   1
    ##         19   0   0   0   0   0   0   0   0   0   0   0   0   0   0   1   0   0
    ##         20   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
    ##         21   0   0   0   0   0   0   0   0   0   0   1   0   0   0   0   0   0
    ##           Reference
    ## Prediction  18  19  20  21
    ##         1    0   0   0   0
    ##         2    0   0   0   0
    ##         3    0   0   0   0
    ##         4    0   0   0   0
    ##         5    0   0   0   0
    ##         6    0   0   0   0
    ##         7    0   0   0   0
    ##         8    0   0   0   0
    ##         9    0   0   0   0
    ##         10   0   0   0   0
    ##         11   1   0   0   0
    ##         12   0   0   0   0
    ##         13   2   0   0   0
    ##         14   1   0   1   0
    ##         15   2   0   0   0
    ##         16   1   0   1   0
    ##         17   0   1   1   0
    ##         18   3   0   0   1
    ##         19   0   0   0   0
    ##         20   1   0   0   0
    ##         21   0   0   0   1
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.5902          
    ##                  95% CI : (0.5652, 0.6148)
    ##     No Information Rate : 0.1157          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.5555          
    ##                                           
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: 1 Class: 2 Class: 3 Class: 4 Class: 5 Class: 6
    ## Sensitivity          0.500000  0.94118  0.64516  0.75000  0.66667  0.64571
    ## Specificity          0.998694  0.99477  0.98945  0.98999  0.98273  0.96210
    ## Pos Pred Value       0.800000  0.66667  0.55556  0.70588  0.72527  0.68485
    ## Neg Pred Value       0.994795  0.99934  0.99272  0.99198  0.97734  0.95514
    ## Prevalence           0.010343  0.01099  0.02004  0.03103  0.06399  0.11312
    ## Detection Rate       0.005171  0.01034  0.01293  0.02327  0.04266  0.07304
    ## Detection Prevalence 0.006464  0.01551  0.02327  0.03297  0.05882  0.10666
    ## Balanced Accuracy    0.749347  0.96797  0.81730  0.87000  0.82470  0.80391
    ##                      Class: 7 Class: 8 Class: 9 Class: 10 Class: 11 Class: 12
    ## Sensitivity           0.67669  0.61453  0.56471   0.62238   0.63964   0.54118
    ## Specificity           0.96252  0.94298  0.94045   0.97436   0.97423   0.95896
    ## Pos Pred Value        0.62937  0.58511  0.53933   0.71200   0.65741   0.43396
    ## Neg Pred Value        0.96937  0.94923  0.94595   0.96203   0.97220   0.97294
    ## Prevalence            0.08597  0.11571  0.10989   0.09244   0.07175   0.05495
    ## Detection Rate        0.05818  0.07111  0.06206   0.05753   0.04590   0.02973
    ## Detection Prevalence  0.09244  0.12153  0.11506   0.08080   0.06981   0.06852
    ## Balanced Accuracy     0.81960  0.77875  0.75258   0.79837   0.80694   0.75007
    ##                      Class: 13 Class: 14 Class: 15 Class: 16 Class: 17
    ## Sensitivity            0.41772   0.52174   0.38462   0.48276   0.48276
    ## Specificity            0.97071   0.96289   0.98448   0.98120   0.99012
    ## Pos Pred Value         0.43421   0.47059   0.52083   0.50000   0.48276
    ## Neg Pred Value         0.96873   0.96955   0.97332   0.97988   0.99012
    ## Prevalence             0.05107   0.05947   0.04202   0.03749   0.01875
    ## Detection Rate         0.02133   0.03103   0.01616   0.01810   0.00905
    ## Detection Prevalence   0.04913   0.06593   0.03103   0.03620   0.01875
    ## Balanced Accuracy      0.69421   0.74231   0.68455   0.73198   0.73644
    ##                      Class: 18 Class: 19 Class: 20 Class: 21
    ## Sensitivity           0.272727 0.0000000 0.0000000 0.5000000
    ## Specificity           0.997396 0.9993532 0.9993523 0.9993528
    ## Pos Pred Value        0.428571 0.0000000 0.0000000 0.5000000
    ## Neg Pred Value        0.994805 0.9993532 0.9980595 0.9993528
    ## Prevalence            0.007111 0.0006464 0.0019392 0.0012928
    ## Detection Rate        0.001939 0.0000000 0.0000000 0.0006464
    ## Detection Prevalence  0.004525 0.0006464 0.0006464 0.0012928
    ## Balanced Accuracy     0.635062 0.4996766 0.4996762 0.7496764

``` r
varImp(bag.model)
```

    ##                                                     Overall
    ## Asset.Turnover                                   1311.88586
    ## Binary.Rating                                     403.94236
    ## Current.Ratio                                    1669.41440
    ## Debt.Equity.Ratio                                1444.78361
    ## Free.Cash.Flow.Per.Share                          919.13640
    ## Gross.Margin                                     1458.33980
    ## Long.term.Debt...Capital                         1590.15072
    ## Net.Profit.Margin                                1394.85674
    ## Operating.Cash.Flow.Per.Share                     940.95661
    ## Rating.Agency_Egan.Jones.Ratings.Company          164.75492
    ## Rating.Agency_Fitch.Ratings                       151.19488
    ## Rating.Agency_Moody.s.Investors.Service           304.28225
    ## Rating.Agency_Standard...Poor.s.Ratings.Services  326.69173
    ## Return.On.Tangible.Equity                        1147.87283
    ## ROA...Return.On.Assets                           1211.18170
    ## ROE...Return.On.Equity                           1225.49649
    ## ROI...Return.On.Investment                       1098.04759
    ## Sector_BusEq                                      199.60001
    ## Sector_Chems                                      108.44646
    ## Sector_Durbl                                       82.44091
    ## Sector_Enrgy                                      168.67990
    ## Sector_Hlth                                       187.86587
    ## Sector_Manuf                                      203.32567
    ## Sector_Money                                       64.59691
    ## Sector_NoDur                                      121.17749
    ## Sector_Other                                      231.00154
    ## Sector_Shops                                      167.03081
    ## Sector_Telcm                                       94.15670
    ## Sector_Utils                                      108.93167
    ## X                                                2010.87979
    ## X.1                                              2190.19838
