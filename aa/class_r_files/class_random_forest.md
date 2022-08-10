# Random Forest

``` r
# Load libraries
library(randomForest)
```

    ## randomForest 4.7-1.1

    ## Type rfNews() to see new features/changes/bug fixes.

``` r
library(caret)
```

    ## Loading required package: ggplot2

    ## 
    ## Attaching package: 'ggplot2'

    ## The following object is masked from 'package:randomForest':
    ## 
    ##     margin

    ## Loading required package: lattice

## Import Data

``` r
# Read training and testing data
train <- read.csv("./../data/classification_data/intermediates/train.csv")
test <- read.csv("./../data/classification_data/intermediates/test.csv")
```

## Model Training

``` r
rf.model <- randomForest(as.factor(Rating)~., data=train)
rf.model
```

    ## 
    ## Call:
    ##  randomForest(formula = as.factor(Rating) ~ ., data = train) 
    ##                Type of random forest: classification
    ##                      Number of trees: 500
    ## No. of variables tried at each split: 5
    ## 
    ##         OOB estimate of  error rate: 42.47%
    ## Confusion matrix:
    ##     1  2  3   4   5   6   7   8   9  10  11  12  13  14  15 16 17 18 19 20 21
    ## 1  42  2  6   4   5   6   4   0   3   0   0   0   0   0   0  0  0  0  0  0  0
    ## 2   4 39  3   2   2   2   0   0   1   0   0   0   0   0   0  0  0  0  0  0  0
    ## 3   4  3 87  10  10  21   3   2   8   6   0   0   0   0   0  0  0  0  0  0  0
    ## 4   4  2 14 126  22  13   9   4   2   1   0   0   0   0   0  0  0  0  0  0  0
    ## 5   5  1  5  27 255  38  18  16   7   6   0   0   0   0   0  0  0  0  0  0  0
    ## 6   3  1 14  10  42 390  73  58  42  16   0   0   0   0   0  0  0  0  0  0  0
    ## 7   2  0  3  12  19  88 324  87  29  13   0   0   0   0   0  0  0  0  0  0  0
    ## 8   1  0  2   2  16  51  75 390 103  20   0   0   0   0   0  0  0  0  0  0  0
    ## 9   2  0  1   1   4  49  30 101 461  88   0   0   0   0   0  0  0  0  0  0  0
    ## 10  0  0  0   0   2  15  14  27 128 373   0   0   0   0   0  0  0  0  0  0  0
    ## 11  0  0  0   0   0   0   0   0   0   1 317  58  30  13  10  3  0  0  0  0  1
    ## 12  0  0  0   0   0   0   0   0   0   0  84 166  62  33  13  9  1  0  0  0  0
    ## 13  0  0  0   0   0   0   0   0   0   0  37  63 167  46  18 11  6  0  0  0  0
    ## 14  0  0  0   0   0   0   0   0   0   0  21  32  57 150  43 23  4  0  0  0  0
    ## 15  0  0  0   0   0   0   0   0   0   0  11  10  19  46 110 36  7  4  1  1  0
    ## 16  0  0  0   0   0   0   0   0   0   0   7  16  18  21  38 97 25  6  0  2  0
    ## 17  0  0  0   0   0   0   0   0   0   0   3   2   7  10   8 25 45 19  0  2  0
    ## 18  0  0  0   0   0   0   0   0   0   0   0   0   0   1   5  6 13 24  1  3  1
    ## 19  0  0  0   0   0   0   0   0   0   0   0   0   1   1   3  1  0  2  0  1  0
    ## 20  0  0  0   0   0   0   0   0   0   0   0   0   0   0   2  2  3  5  2  0  1
    ## 21  0  0  0   0   0   0   0   0   0   0   1   0   0   0   0  0  0  3  0  2  3
    ##    class.error
    ## 1    0.4166667
    ## 2    0.2641509
    ## 3    0.4350649
    ## 4    0.3604061
    ## 5    0.3253968
    ## 6    0.3990755
    ## 7    0.4384749
    ## 8    0.4090909
    ## 9    0.3744912
    ## 10   0.3327370
    ## 11   0.2678984
    ## 12   0.5489130
    ## 13   0.5201149
    ## 14   0.5454545
    ## 15   0.5510204
    ## 16   0.5782609
    ## 17   0.6280992
    ## 18   0.5555556
    ## 19   1.0000000
    ## 20   1.0000000
    ## 21   0.6666667

## Model Validation

``` r
result <- predict(rf.model, test, type="class")

confusionMatrix(as.factor(result), as.factor(test$Rating))
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17
    ##         1   11   0   0   1   1   0   2   0   0   0   0   0   0   0   0   0   0
    ##         2    3  16   1   2   1   0   1   0   1   0   0   0   0   0   0   0   0
    ##         3    1   0  17   3   6   5   2   0   0   0   0   0   0   0   0   0   0
    ##         4    0   0   2  37  10   3   2   2   1   0   0   0   0   0   0   0   0
    ##         5    1   0   3   1  66  13   3   1   0   0   0   0   0   0   0   0   0
    ##         6    0   0   6   1   4 114  14  13  12   3   0   0   0   0   0   0   0
    ##         7    0   0   1   2   3  17  90  19   8   4   0   0   0   0   0   0   0
    ##         8    0   0   0   0   3  11  15 117  37   8   0   0   0   0   0   0   0
    ##         9    0   1   1   1   2   9   2  24  93  35   0   0   0   0   0   0   0
    ##         10   0   0   0   0   3   3   2   3  18  93   0   0   0   0   0   0   0
    ##         11   0   0   0   0   0   0   0   0   0   0  77  20  12   8   2   3   1
    ##         12   0   0   0   0   0   0   0   0   0   0  22  41  11  11   3   5   1
    ##         13   0   0   0   0   0   0   0   0   0   0   6  13  37  11   8   3   4
    ##         14   0   0   0   0   0   0   0   0   0   0   1   8  13  46  14   4   2
    ##         15   0   0   0   0   0   0   0   0   0   0   1   0   2  13  28   6   6
    ##         16   0   0   0   0   0   0   0   0   0   0   3   2   3   3   7  29   3
    ##         17   0   0   0   0   0   0   0   0   0   0   0   1   1   0   3   6  10
    ##         18   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   2   2
    ##         19   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
    ##         20   0   0   0   0   0   0   0   0   0   0   1   0   0   0   0   0   0
    ##         21   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
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
    ##         13   3   0   0   0
    ##         14   1   0   2   0
    ##         15   1   0   0   0
    ##         16   1   0   0   0
    ##         17   1   1   1   0
    ##         18   2   0   0   1
    ##         19   0   0   0   0
    ##         20   1   0   0   0
    ##         21   0   0   0   1
    ## 
    ## Overall Statistics
    ##                                          
    ##                Accuracy : 0.5979         
    ##                  95% CI : (0.573, 0.6225)
    ##     No Information Rate : 0.1157         
    ##     P-Value [Acc > NIR] : < 2.2e-16      
    ##                                          
    ##                   Kappa : 0.564          
    ##                                          
    ##  Mcnemar's Test P-Value : NA             
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: 1 Class: 2 Class: 3 Class: 4 Class: 5 Class: 6
    ## Sensitivity          0.687500  0.94118  0.54839  0.77083  0.66667  0.65143
    ## Specificity          0.997387  0.99412  0.98879  0.98666  0.98481  0.96137
    ## Pos Pred Value       0.733333  0.64000  0.50000  0.64912  0.75000  0.68263
    ## Neg Pred Value       0.996736  0.99934  0.99075  0.99262  0.97738  0.95580
    ## Prevalence           0.010343  0.01099  0.02004  0.03103  0.06399  0.11312
    ## Detection Rate       0.007111  0.01034  0.01099  0.02392  0.04266  0.07369
    ## Detection Prevalence 0.009696  0.01616  0.02198  0.03685  0.05688  0.10795
    ## Balanced Accuracy    0.842444  0.96765  0.76859  0.87875  0.82574  0.80640
    ##                      Class: 7 Class: 8 Class: 9 Class: 10 Class: 11 Class: 12
    ## Sensitivity           0.67669  0.65363  0.54706   0.65035   0.69369   0.48235
    ## Specificity           0.96181  0.94591  0.94553   0.97934   0.96727   0.96375
    ## Pos Pred Value        0.62500  0.61257  0.55357   0.76230   0.62097   0.43617
    ## Neg Pred Value        0.96935  0.95428  0.94416   0.96491   0.97611   0.96972
    ## Prevalence            0.08597  0.11571  0.10989   0.09244   0.07175   0.05495
    ## Detection Rate        0.05818  0.07563  0.06012   0.06012   0.04977   0.02650
    ## Detection Prevalence  0.09308  0.12346  0.10860   0.07886   0.08016   0.06076
    ## Balanced Accuracy     0.81925  0.79977  0.74630   0.81485   0.83048   0.72305
    ##                      Class: 13 Class: 14 Class: 15 Class: 16 Class: 17
    ## Sensitivity            0.46835   0.50000   0.43077   0.50000  0.344828
    ## Specificity            0.96730   0.96907   0.98043   0.98522  0.990777
    ## Pos Pred Value         0.43529   0.50549   0.49123   0.56863  0.416667
    ## Neg Pred Value         0.97127   0.96841   0.97517   0.98061  0.987525
    ## Prevalence             0.05107   0.05947   0.04202   0.03749  0.018746
    ## Detection Rate         0.02392   0.02973   0.01810   0.01875  0.006464
    ## Detection Prevalence   0.05495   0.05882   0.03685   0.03297  0.015514
    ## Balanced Accuracy      0.71783   0.73454   0.70560   0.74261  0.667802
    ##                      Class: 18 Class: 19 Class: 20 Class: 21
    ## Sensitivity           0.181818 0.0000000  0.000000 0.5000000
    ## Specificity           0.996745 1.0000000  0.998705 1.0000000
    ## Pos Pred Value        0.285714       NaN  0.000000 1.0000000
    ## Neg Pred Value        0.994156 0.9993536  0.998058 0.9993532
    ## Prevalence            0.007111 0.0006464  0.001939 0.0012928
    ## Detection Rate        0.001293 0.0000000  0.000000 0.0006464
    ## Detection Prevalence  0.004525 0.0000000  0.001293 0.0006464
    ## Balanced Accuracy     0.589281 0.5000000  0.499352 0.7500000

``` r
varImp(rf.model)
```

    ##                                                    Overall
    ## X.1                                              400.79060
    ## X                                                402.18946
    ## Rating.Agency_Egan.Jones.Ratings.Company          95.33565
    ## Rating.Agency_Fitch.Ratings                       36.80788
    ## Rating.Agency_Moody.s.Investors.Service           78.20522
    ## Rating.Agency_Standard...Poor.s.Ratings.Services  91.93385
    ## Binary.Rating                                    361.01423
    ## Sector_BusEq                                      39.60563
    ## Sector_Chems                                      20.06729
    ## Sector_Durbl                                      13.78857
    ## Sector_Enrgy                                      30.33805
    ## Sector_Hlth                                       32.07982
    ## Sector_Manuf                                      38.57074
    ## Sector_Money                                      12.12231
    ## Sector_NoDur                                      23.63366
    ## Sector_Other                                      43.98253
    ## Sector_Shops                                      33.47263
    ## Sector_Telcm                                      20.51995
    ## Sector_Utils                                      23.51925
    ## Current.Ratio                                    317.53129
    ## Long.term.Debt...Capital                         307.85088
    ## Debt.Equity.Ratio                                294.19258
    ## Gross.Margin                                     313.76846
    ## Net.Profit.Margin                                297.57179
    ## Asset.Turnover                                   293.69182
    ## ROE...Return.On.Equity                           276.48002
    ## Return.On.Tangible.Equity                        284.44509
    ## ROA...Return.On.Assets                           283.14404
    ## ROI...Return.On.Investment                       284.56018
    ## Operating.Cash.Flow.Per.Share                    247.70634
    ## Free.Cash.Flow.Per.Share                         244.75940
