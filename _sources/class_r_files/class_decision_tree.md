# Decision Tree

``` r
# Load libraries
library(rpart)
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
dtree.model <- rpart(Rating~., data=train, method = 'class')
summary(dtree.model)
```

    ## Call:
    ## rpart(formula = Rating ~ ., data = train, method = "class")
    ##   n= 6198 
    ## 
    ##           CP nsplit rel error    xerror        xstd
    ## 1 0.07928951      0 1.0000000 1.0000000 0.004666292
    ## 2 0.01336752      1 0.9207105 0.9207105 0.005641474
    ## 3 0.01263505      2 0.9073430 0.9130196 0.005717807
    ## 4 0.01135323      3 0.8947079 0.8980040 0.005859282
    ## 5 0.01000000      4 0.8833547 0.8895807 0.005934487
    ## 
    ## Variable importance
    ##                                    Binary.Rating 
    ##                                               38 
    ##                           ROE...Return.On.Equity 
    ##                                               12 
    ##                       ROI...Return.On.Investment 
    ##                                               12 
    ##                           ROA...Return.On.Assets 
    ##                                               11 
    ##                                Net.Profit.Margin 
    ##                                               10 
    ##                         Long.term.Debt...Capital 
    ##                                                8 
    ##         Rating.Agency_Egan.Jones.Ratings.Company 
    ##                                                4 
    ##                                    Current.Ratio 
    ##                                                2 
    ## Rating.Agency_Standard...Poor.s.Ratings.Services 
    ##                                                2 
    ##                                Debt.Equity.Ratio 
    ##                                                1 
    ## 
    ## Node number 1: 6198 observations,    complexity param=0.07928951
    ##   predicted class=9   expected loss=0.8810907  P(node) =1
    ##     class counts:    72    53   154   197   378   649   577   660   737   559   433   368   348   330   245   230   121    54     9    15     9
    ##    probabilities: 0.012 0.009 0.025 0.032 0.061 0.105 0.093 0.106 0.119 0.090 0.070 0.059 0.056 0.053 0.040 0.037 0.020 0.009 0.001 0.002 0.001 
    ##   left son=2 (4036 obs) right son=3 (2162 obs)
    ##   Primary splits:
    ##       Binary.Rating              < 0.5      to the right, improve=401.17240, (0 missing)
    ##       ROI...Return.On.Investment < 2.9706   to the right, improve= 61.76188, (0 missing)
    ##       Net.Profit.Margin          < 4.30595  to the right, improve= 60.56801, (0 missing)
    ##       ROE...Return.On.Equity     < 4.42715  to the right, improve= 58.08825, (0 missing)
    ##       ROA...Return.On.Assets     < 1.23085  to the right, improve= 57.45164, (0 missing)
    ##   Surrogate splits:
    ##       ROI...Return.On.Investment < 2.23125  to the right, agree=0.739, adj=0.253, (0 split)
    ##       ROA...Return.On.Assets     < 1.2463   to the right, agree=0.733, adj=0.235, (0 split)
    ##       ROE...Return.On.Equity     < 4.2536   to the right, agree=0.731, adj=0.228, (0 split)
    ##       Net.Profit.Margin          < 1.76035  to the right, agree=0.727, adj=0.218, (0 split)
    ##       Long.term.Debt...Capital   < 0.62395  to the left,  agree=0.713, adj=0.177, (0 split)
    ## 
    ## Node number 2: 4036 observations,    complexity param=0.01336752
    ##   predicted class=9   expected loss=0.8173935  P(node) =0.6511778
    ##     class counts:    72    53   154   197   378   649   577   660   737   559     0     0     0     0     0     0     0     0     0     0     0
    ##    probabilities: 0.018 0.013 0.038 0.049 0.094 0.161 0.143 0.164 0.183 0.139 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 
    ##   left son=4 (1710 obs) right son=5 (2326 obs)
    ##   Primary splits:
    ##       Rating.Agency_Egan.Jones.Ratings.Company < 0.5      to the right, improve=37.26143, (0 missing)
    ##       ROI...Return.On.Investment               < 10.79385 to the right, improve=33.72029, (0 missing)
    ##       ROA...Return.On.Assets                   < 5.7154   to the right, improve=30.27198, (0 missing)
    ##       Current.Ratio                            < 1.56795  to the left,  improve=25.63573, (0 missing)
    ##       Net.Profit.Margin                        < 6.4634   to the right, improve=23.18474, (0 missing)
    ##   Surrogate splits:
    ##       Rating.Agency_Standard...Poor.s.Ratings.Services < 0.5      to the left,  agree=0.757, adj=0.427, (0 split)
    ##       X.1                                              < 2698     to the right, agree=0.610, adj=0.078, (0 split)
    ##       X                                                < 2698     to the right, agree=0.610, adj=0.078, (0 split)
    ##       Rating.Agency_Moody.s.Investors.Service          < 0.5      to the left,  agree=0.600, adj=0.055, (0 split)
    ##       Sector_Shops                                     < 0.5      to the right, agree=0.585, adj=0.020, (0 split)
    ## 
    ## Node number 3: 2162 observations,    complexity param=0.01263505
    ##   predicted class=11  expected loss=0.7997225  P(node) =0.3488222
    ##     class counts:     0     0     0     0     0     0     0     0     0     0   433   368   348   330   245   230   121    54     9    15     9
    ##    probabilities: 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.200 0.170 0.161 0.153 0.113 0.106 0.056 0.025 0.004 0.007 0.004 
    ##   left son=6 (1571 obs) right son=7 (591 obs)
    ##   Primary splits:
    ##       ROE...Return.On.Equity     < -1.9733  to the right, improve=32.51677, (0 missing)
    ##       Net.Profit.Margin          < 1.75235  to the right, improve=30.08367, (0 missing)
    ##       ROA...Return.On.Assets     < 0.05915  to the right, improve=27.49627, (0 missing)
    ##       ROI...Return.On.Investment < -0.60695 to the right, improve=24.78331, (0 missing)
    ##       Long.term.Debt...Capital   < 0.73535  to the left,  improve=22.88975, (0 missing)
    ##   Surrogate splits:
    ##       ROI...Return.On.Investment < -0.60695 to the right, agree=0.909, adj=0.668, (0 split)
    ##       ROA...Return.On.Assets     < -0.7403  to the right, agree=0.905, adj=0.653, (0 split)
    ##       Net.Profit.Margin          < -1.18545 to the right, agree=0.884, adj=0.577, (0 split)
    ##       Debt.Equity.Ratio          < 0.0013   to the right, agree=0.766, adj=0.146, (0 split)
    ##       Long.term.Debt...Capital   < 0.9892   to the left,  agree=0.765, adj=0.139, (0 split)
    ## 
    ## Node number 4: 1710 observations
    ##   predicted class=7   expected loss=0.8146199  P(node) =0.2758955
    ##     class counts:     2    34    62   140   269   267   317   228   244   147     0     0     0     0     0     0     0     0     0     0     0
    ##    probabilities: 0.001 0.020 0.036 0.082 0.157 0.156 0.185 0.133 0.143 0.086 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 
    ## 
    ## Node number 5: 2326 observations,    complexity param=0.01135323
    ##   predicted class=9   expected loss=0.7880482  P(node) =0.3752823
    ##     class counts:    70    19    92    57   109   382   260   432   493   412     0     0     0     0     0     0     0     0     0     0     0
    ##    probabilities: 0.030 0.008 0.040 0.025 0.047 0.164 0.112 0.186 0.212 0.177 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 
    ##   left son=10 (1344 obs) right son=11 (982 obs)
    ##   Primary splits:
    ##       Current.Ratio                                    < 1.54875  to the left,  improve=17.31650, (0 missing)
    ##       Rating.Agency_Standard...Poor.s.Ratings.Services < 0.5      to the right, improve=15.34410, (0 missing)
    ##       ROE...Return.On.Equity                           < 7.4847   to the right, improve=14.90334, (0 missing)
    ##       Net.Profit.Margin                                < 5.1835   to the right, improve=13.66931, (0 missing)
    ##       ROI...Return.On.Investment                       < 10.95775 to the right, improve=13.61317, (0 missing)
    ##   Surrogate splits:
    ##       Long.term.Debt...Capital  < 0.36225  to the right, agree=0.666, adj=0.209, (0 split)
    ##       Debt.Equity.Ratio         < 0.53465  to the right, agree=0.664, adj=0.204, (0 split)
    ##       ROA...Return.On.Assets    < 9.69045  to the left,  agree=0.640, adj=0.148, (0 split)
    ##       Return.On.Tangible.Equity < 15.0308  to the left,  agree=0.632, adj=0.127, (0 split)
    ##       Sector_Manuf              < 0.5      to the left,  agree=0.618, adj=0.095, (0 split)
    ## 
    ## Node number 6: 1571 observations
    ##   predicted class=11  expected loss=0.7542966  P(node) =0.2534689
    ##     class counts:     0     0     0     0     0     0     0     0     0     0   386   311   287   237   140   114    53    22     7    11     3
    ##    probabilities: 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.246 0.198 0.183 0.151 0.089 0.073 0.034 0.014 0.004 0.007 0.002 
    ## 
    ## Node number 7: 591 observations
    ##   predicted class=16  expected loss=0.8037225  P(node) =0.09535334
    ##     class counts:     0     0     0     0     0     0     0     0     0     0    47    57    61    93   105   116    68    32     2     4     6
    ##    probabilities: 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.080 0.096 0.103 0.157 0.178 0.196 0.115 0.054 0.003 0.007 0.010 
    ## 
    ## Node number 10: 1344 observations
    ##   predicted class=8   expected loss=0.7745536  P(node) =0.2168441
    ##     class counts:    29    13    50    38    50   255   179   303   241   186     0     0     0     0     0     0     0     0     0     0     0
    ##    probabilities: 0.022 0.010 0.037 0.028 0.037 0.190 0.133 0.225 0.179 0.138 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 
    ## 
    ## Node number 11: 982 observations
    ##   predicted class=9   expected loss=0.7433809  P(node) =0.1584382
    ##     class counts:    41     6    42    19    59   127    81   129   252   226     0     0     0     0     0     0     0     0     0     0     0
    ##    probabilities: 0.042 0.006 0.043 0.019 0.060 0.129 0.082 0.131 0.257 0.230 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000

## Model Validation

``` r
result <- predict(dtree.model, test, type="class")

confusionMatrix(as.factor(result), as.factor(test$Rating))
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17
    ##         1    0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
    ##         2    0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
    ##         3    0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
    ##         4    0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
    ##         5    0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
    ##         6    0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
    ##         7    3  12  15  41  70  75  73  67  60  43   0   0   0   0   0   0   0
    ##         8    4   4   9   5  10  74  40  82  55  55   0   0   0   0   0   0   0
    ##         9    9   1   7   2  19  26  20  30  55  45   0   0   0   0   0   0   0
    ##         10   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
    ##         11   0   0   0   0   0   0   0   0   0   0 100  78  62  70  40  32  16
    ##         12   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
    ##         13   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
    ##         14   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
    ##         15   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
    ##         16   0   0   0   0   0   0   0   0   0   0  11   7  17  22  25  26  13
    ##         17   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
    ##         18   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
    ##         19   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
    ##         20   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
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
    ##         11   6   0   3   1
    ##         12   0   0   0   0
    ##         13   0   0   0   0
    ##         14   0   0   0   0
    ##         15   0   0   0   0
    ##         16   5   1   0   1
    ##         17   0   0   0   0
    ##         18   0   0   0   0
    ##         19   0   0   0   0
    ##         20   0   0   0   0
    ##         21   0   0   0   0
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.2172          
    ##                  95% CI : (0.1969, 0.2386)
    ##     No Information Rate : 0.1157          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.1416          
    ##                                           
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: 1 Class: 2 Class: 3 Class: 4 Class: 5 Class: 6
    ## Sensitivity           0.00000  0.00000  0.00000  0.00000  0.00000   0.0000
    ## Specificity           1.00000  1.00000  1.00000  1.00000  1.00000   1.0000
    ## Pos Pred Value            NaN      NaN      NaN      NaN      NaN      NaN
    ## Neg Pred Value        0.98966  0.98901  0.97996  0.96897  0.93601   0.8869
    ## Prevalence            0.01034  0.01099  0.02004  0.03103  0.06399   0.1131
    ## Detection Rate        0.00000  0.00000  0.00000  0.00000  0.00000   0.0000
    ## Detection Prevalence  0.00000  0.00000  0.00000  0.00000  0.00000   0.0000
    ## Balanced Accuracy     0.50000  0.50000  0.50000  0.50000  0.50000   0.5000
    ##                      Class: 7 Class: 8 Class: 9 Class: 10 Class: 11 Class: 12
    ## Sensitivity           0.54887  0.45810  0.32353   0.00000   0.90090   0.00000
    ## Specificity           0.72702  0.81287  0.88453   1.00000   0.78552   1.00000
    ## Pos Pred Value        0.15904  0.24260  0.25701       NaN   0.24510       NaN
    ## Neg Pred Value        0.94485  0.91977  0.91373   0.90756   0.99034   0.94505
    ## Prevalence            0.08597  0.11571  0.10989   0.09244   0.07175   0.05495
    ## Detection Rate        0.04719  0.05301  0.03555   0.00000   0.06464   0.00000
    ## Detection Prevalence  0.29670  0.21849  0.13833   0.00000   0.26374   0.00000
    ## Balanced Accuracy     0.63794  0.63548  0.60403   0.50000   0.84321   0.50000
    ##                      Class: 13 Class: 14 Class: 15 Class: 16 Class: 17
    ## Sensitivity            0.00000   0.00000   0.00000   0.44828   0.00000
    ## Specificity            1.00000   1.00000   1.00000   0.93150   1.00000
    ## Pos Pred Value             NaN       NaN       NaN   0.20312       NaN
    ## Neg Pred Value         0.94893   0.94053   0.95798   0.97745   0.98125
    ## Prevalence             0.05107   0.05947   0.04202   0.03749   0.01875
    ## Detection Rate         0.00000   0.00000   0.00000   0.01681   0.00000
    ## Detection Prevalence   0.00000   0.00000   0.00000   0.08274   0.00000
    ## Balanced Accuracy      0.50000   0.50000   0.50000   0.68989   0.50000
    ##                      Class: 18 Class: 19 Class: 20 Class: 21
    ## Sensitivity           0.000000 0.0000000  0.000000  0.000000
    ## Specificity           1.000000 1.0000000  1.000000  1.000000
    ## Pos Pred Value             NaN       NaN       NaN       NaN
    ## Neg Pred Value        0.992889 0.9993536  0.998061  0.998707
    ## Prevalence            0.007111 0.0006464  0.001939  0.001293
    ## Detection Rate        0.000000 0.0000000  0.000000  0.000000
    ## Detection Prevalence  0.000000 0.0000000  0.000000  0.000000
    ## Balanced Accuracy     0.500000 0.5000000  0.500000  0.500000

``` r
varImp(dtree.model)
```

    ##                                                    Overall
    ## Binary.Rating                                    401.17243
    ## Current.Ratio                                     42.95223
    ## Long.term.Debt...Capital                          22.88975
    ## Net.Profit.Margin                                127.50573
    ## Rating.Agency_Egan.Jones.Ratings.Company          37.26143
    ## Rating.Agency_Standard...Poor.s.Ratings.Services  15.34410
    ## ROA...Return.On.Assets                           115.21988
    ## ROE...Return.On.Equity                           105.50835
    ## ROI...Return.On.Investment                       133.87866
    ## X.1                                                0.00000
    ## X                                                  0.00000
    ## Rating.Agency_Fitch.Ratings                        0.00000
    ## Rating.Agency_Moody.s.Investors.Service            0.00000
    ## Sector_BusEq                                       0.00000
    ## Sector_Chems                                       0.00000
    ## Sector_Durbl                                       0.00000
    ## Sector_Enrgy                                       0.00000
    ## Sector_Hlth                                        0.00000
    ## Sector_Manuf                                       0.00000
    ## Sector_Money                                       0.00000
    ## Sector_NoDur                                       0.00000
    ## Sector_Other                                       0.00000
    ## Sector_Shops                                       0.00000
    ## Sector_Telcm                                       0.00000
    ## Sector_Utils                                       0.00000
    ## Debt.Equity.Ratio                                  0.00000
    ## Gross.Margin                                       0.00000
    ## Asset.Turnover                                     0.00000
    ## Return.On.Tangible.Equity                          0.00000
    ## Operating.Cash.Flow.Per.Share                      0.00000
    ## Free.Cash.Flow.Per.Share                           0.00000