# Naive Bayes

``` r
# Load libraries
library(e1071)
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
nb.model <- naiveBayes(Rating~., data=train)
nb.model
```

    ## 
    ## Naive Bayes Classifier for Discrete Predictors
    ## 
    ## Call:
    ## naiveBayes.default(x = X, y = Y, laplace = laplace)
    ## 
    ## A-priori probabilities:
    ## Y
    ##           1           2           3           4           5           6 
    ## 0.011293966 0.008873830 0.023071959 0.031139077 0.064375605 0.105679251 
    ##           7           8           9          10          11          12 
    ## 0.091158438 0.109390126 0.116005163 0.090029042 0.070022588 0.059373992 
    ##          13          14          15          16          17          18 
    ## 0.054211036 0.053888351 0.041787673 0.039044853 0.017747661 0.008067118 
    ##          19          20          21 
    ## 0.001452081 0.002258793 0.001129397 
    ## 
    ## Conditional probabilities:
    ##     X.1
    ## Y        [,1]     [,2]
    ##   1  3609.943 2653.199
    ##   2  4039.782 1737.441
    ##   3  4228.252 2266.765
    ##   4  3953.311 1743.602
    ##   5  3947.361 1822.377
    ##   6  3926.089 2208.589
    ##   7  3872.577 2034.525
    ##   8  3627.249 2190.780
    ##   9  3693.605 2255.794
    ##   10 3980.176 2384.937
    ##   11 4305.818 2219.841
    ##   12 4051.535 2329.974
    ##   13 3891.506 2405.976
    ##   14 3925.605 2362.739
    ##   15 3631.583 2402.349
    ##   16 3938.219 2355.287
    ##   17 4184.391 2323.009
    ##   18 2943.160 2174.502
    ##   19 3564.111 2827.873
    ##   20 3993.786 2981.799
    ##   21 2717.571 2582.720
    ## 
    ##     X
    ## Y        [,1]     [,2]
    ##   1  3609.943 2653.199
    ##   2  4039.782 1737.441
    ##   3  4228.252 2266.765
    ##   4  3953.311 1743.602
    ##   5  3947.361 1822.377
    ##   6  3926.089 2208.589
    ##   7  3872.577 2034.525
    ##   8  3627.249 2190.780
    ##   9  3693.605 2255.794
    ##   10 3980.176 2384.937
    ##   11 4305.818 2219.841
    ##   12 4051.535 2329.974
    ##   13 3891.506 2405.976
    ##   14 3925.605 2362.739
    ##   15 3631.583 2402.349
    ##   16 3938.219 2355.287
    ##   17 4184.391 2323.009
    ##   18 2943.160 2174.502
    ##   19 3564.111 2827.873
    ##   20 3993.786 2981.799
    ##   21 2717.571 2582.720
    ## 
    ##     Rating.Agency_Egan.Jones.Ratings.Company
    ## Y          [,1]      [,2]
    ##   1  0.04285714 0.2039973
    ##   2  0.65454545 0.4798990
    ##   3  0.40559441 0.4927326
    ##   4  0.73575130 0.4420791
    ##   5  0.70927318 0.4546678
    ##   6  0.43053435 0.4955294
    ##   7  0.54336283 0.4985575
    ##   8  0.36135693 0.4807484
    ##   9  0.35326843 0.4783179
    ##   10 0.26881720 0.4437425
    ##   11 0.39400922 0.4892008
    ##   12 0.23641304 0.4254572
    ##   13 0.17857143 0.3835643
    ##   14 0.20958084 0.4076201
    ##   15 0.18918919 0.3924171
    ##   16 0.19008264 0.3931794
    ##   17 0.15454545 0.3631252
    ##   18 0.28000000 0.4535574
    ##   19 0.11111111 0.3333333
    ##   20 0.07142857 0.2672612
    ##   21 0.42857143 0.5345225
    ## 
    ##     Rating.Agency_Fitch.Ratings
    ## Y          [,1]      [,2]
    ##   1  0.00000000 0.0000000
    ##   2  0.00000000 0.0000000
    ##   3  0.01398601 0.1178453
    ##   4  0.03626943 0.1874460
    ##   5  0.03007519 0.1710087
    ##   6  0.06259542 0.2424190
    ##   7  0.03539823 0.1849479
    ##   8  0.10471976 0.3064180
    ##   9  0.07093185 0.2568897
    ##   10 0.11827957 0.3232286
    ##   11 0.06451613 0.2459536
    ##   12 0.05978261 0.2374064
    ##   13 0.03273810 0.1782157
    ##   14 0.02694611 0.1621689
    ##   15 0.05019305 0.2187658
    ##   16 0.04545455 0.2087306
    ##   17 0.06363636 0.2452212
    ##   18 0.14000000 0.3505098
    ##   19 0.00000000 0.0000000
    ##   20 0.14285714 0.3631365
    ##   21 0.14285714 0.3779645
    ## 
    ##     Rating.Agency_Moody.s.Investors.Service
    ## Y          [,1]      [,2]
    ##   1  0.08571429 0.2819630
    ##   2  0.21818182 0.4168182
    ##   3  0.07692308 0.2674060
    ##   4  0.05181347 0.2222267
    ##   5  0.12030075 0.3257213
    ##   6  0.08702290 0.2820840
    ##   7  0.19115044 0.3935557
    ##   8  0.23451327 0.4240070
    ##   9  0.21835883 0.4134198
    ##   10 0.22580645 0.4184875
    ##   11 0.17281106 0.3785202
    ##   12 0.28532609 0.4521844
    ##   13 0.28869048 0.4538296
    ##   14 0.31137725 0.4637514
    ##   15 0.30115830 0.4596496
    ##   16 0.36363636 0.4820427
    ##   17 0.38181818 0.4880558
    ##   18 0.22000000 0.4184520
    ##   19 0.66666667 0.5000000
    ##   20 0.42857143 0.5135526
    ##   21 0.00000000 0.0000000
    ## 
    ##     Rating.Agency_Standard...Poor.s.Ratings.Services
    ## Y         [,1]      [,2]
    ##   1  0.8714286 0.3371418
    ##   2  0.1272727 0.3363500
    ##   3  0.5034965 0.5017452
    ##   4  0.1761658 0.3819520
    ##   5  0.1403509 0.3477868
    ##   6  0.4198473 0.4939109
    ##   7  0.2300885 0.4212622
    ##   8  0.2994100 0.4583378
    ##   9  0.3574409 0.4795798
    ##   10 0.3870968 0.4875231
    ##   11 0.3686636 0.4829993
    ##   12 0.4184783 0.4939811
    ##   13 0.5000000 0.5007457
    ##   14 0.4520958 0.4984466
    ##   15 0.4594595 0.4993186
    ##   16 0.4008264 0.4910816
    ##   17 0.4000000 0.4921401
    ##   18 0.3600000 0.4848732
    ##   19 0.2222222 0.4409586
    ##   20 0.3571429 0.4972452
    ##   21 0.4285714 0.5345225
    ## 
    ##     Binary.Rating
    ## Y    [,1] [,2]
    ##   1     1    0
    ##   2     1    0
    ##   3     1    0
    ##   4     1    0
    ##   5     1    0
    ##   6     1    0
    ##   7     1    0
    ##   8     1    0
    ##   9     1    0
    ##   10    1    0
    ##   11    0    0
    ##   12    0    0
    ##   13    0    0
    ##   14    0    0
    ##   15    0    0
    ##   16    0    0
    ##   17    0    0
    ##   18    0    0
    ##   19    0    0
    ##   20    0    0
    ##   21    0    0
    ## 
    ##     Sector_BusEq
    ## Y          [,1]      [,2]
    ##   1  0.27142857 0.4479075
    ##   2  0.20000000 0.4036867
    ##   3  0.22377622 0.4182388
    ##   4  0.16062176 0.3681367
    ##   5  0.14786967 0.3554164
    ##   6  0.10992366 0.3130336
    ##   7  0.10973451 0.3128355
    ##   8  0.08259587 0.2754736
    ##   9  0.12239221 0.3279664
    ##   10 0.16308244 0.3697724
    ##   11 0.14055300 0.3479610
    ##   12 0.14130435 0.3488095
    ##   13 0.11904762 0.3243275
    ##   14 0.09281437 0.2906075
    ##   15 0.10810811 0.3111181
    ##   16 0.08264463 0.2759150
    ##   17 0.05454545 0.2281302
    ##   18 0.14000000 0.3505098
    ##   19 0.00000000 0.0000000
    ##   20 0.00000000 0.0000000
    ##   21 0.00000000 0.0000000
    ## 
    ##     Sector_Chems
    ## Y           [,1]       [,2]
    ##   1  0.000000000 0.00000000
    ##   2  0.000000000 0.00000000
    ##   3  0.020979021 0.14381774
    ##   4  0.000000000 0.00000000
    ##   5  0.042606516 0.20222189
    ##   6  0.053435115 0.22507142
    ##   7  0.003539823 0.05944364
    ##   8  0.025073746 0.15646457
    ##   9  0.084840056 0.27883751
    ##   10 0.062724014 0.24268346
    ##   11 0.032258065 0.17688860
    ##   12 0.010869565 0.10383021
    ##   13 0.038690476 0.19314385
    ##   14 0.086826347 0.28200290
    ##   15 0.019305019 0.13786122
    ##   16 0.024793388 0.15581721
    ##   17 0.009090909 0.09534626
    ##   18 0.020000000 0.14142136
    ##   19 0.000000000 0.00000000
    ##   20 0.000000000 0.00000000
    ##   21 0.000000000 0.00000000
    ## 
    ##     Sector_Durbl
    ## Y           [,1]       [,2]
    ##   1  0.000000000 0.00000000
    ##   2  0.000000000 0.00000000
    ##   3  0.000000000 0.00000000
    ##   4  0.025906736 0.15927025
    ##   5  0.052631579 0.22357723
    ##   6  0.027480916 0.16360495
    ##   7  0.012389381 0.11071395
    ##   8  0.025073746 0.15646457
    ##   9  0.019471488 0.13827126
    ##   10 0.025089606 0.15653764
    ##   11 0.029953917 0.17065692
    ##   12 0.046195652 0.21019437
    ##   13 0.020833333 0.14303915
    ##   14 0.029940120 0.17067785
    ##   15 0.108108108 0.31111809
    ##   16 0.070247934 0.25609407
    ##   17 0.009090909 0.09534626
    ##   18 0.020000000 0.14142136
    ##   19 0.000000000 0.00000000
    ##   20 0.000000000 0.00000000
    ##   21 0.000000000 0.00000000
    ## 
    ##     Sector_Enrgy
    ## Y          [,1]      [,2]
    ##   1  0.22857143 0.4229444
    ##   2  0.43636364 0.5005048
    ##   3  0.11888112 0.3247862
    ##   4  0.02072539 0.1428340
    ##   5  0.05513784 0.2285357
    ##   6  0.09618321 0.2950677
    ##   7  0.07964602 0.2709843
    ##   8  0.11356932 0.3175217
    ##   9  0.06119611 0.2398565
    ##   10 0.04301075 0.2030634
    ##   11 0.08064516 0.2726036
    ##   12 0.06521739 0.2472452
    ##   13 0.04761905 0.2132765
    ##   14 0.07185629 0.2586373
    ##   15 0.17760618 0.3829208
    ##   16 0.11983471 0.3254412
    ##   17 0.20909091 0.4085206
    ##   18 0.02000000 0.1414214
    ##   19 0.44444444 0.5270463
    ##   20 0.21428571 0.4258153
    ##   21 0.14285714 0.3779645
    ## 
    ##     Sector_Hlth
    ## Y          [,1]      [,2]
    ##   1  0.37142857 0.4866755
    ##   2  0.20000000 0.4036867
    ##   3  0.11188811 0.3163368
    ##   4  0.22797927 0.4206203
    ##   5  0.17293233 0.3786636
    ##   6  0.05343511 0.2250714
    ##   7  0.09203540 0.2893321
    ##   8  0.07817109 0.2686388
    ##   9  0.03616134 0.1868214
    ##   10 0.04301075 0.2030634
    ##   11 0.06682028 0.2499987
    ##   12 0.03804348 0.1915617
    ##   13 0.03869048 0.1931439
    ##   14 0.08083832 0.2729956
    ##   15 0.08880309 0.2850101
    ##   16 0.11983471 0.3254412
    ##   17 0.18181818 0.3874598
    ##   18 0.08000000 0.2740475
    ##   19 0.00000000 0.0000000
    ##   20 0.00000000 0.0000000
    ##   21 0.00000000 0.0000000
    ## 
    ##     Sector_Manuf
    ## Y          [,1]      [,2]
    ##   1  0.01428571 0.1195229
    ##   2  0.00000000 0.0000000
    ##   3  0.22377622 0.4182388
    ##   4  0.07772021 0.2684271
    ##   5  0.13784461 0.3451696
    ##   6  0.10534351 0.3072301
    ##   7  0.09203540 0.2893321
    ##   8  0.10914454 0.3120507
    ##   9  0.14742698 0.3547779
    ##   10 0.13261649 0.3394641
    ##   11 0.17511521 0.3805042
    ##   12 0.17663043 0.3818748
    ##   13 0.18452381 0.3884893
    ##   14 0.11976048 0.3251684
    ##   15 0.07722008 0.2674572
    ##   16 0.07438017 0.2629324
    ##   17 0.15454545 0.3631252
    ##   18 0.04000000 0.1979487
    ##   19 0.00000000 0.0000000
    ##   20 0.00000000 0.0000000
    ##   21 0.00000000 0.0000000
    ## 
    ##     Sector_Money
    ## Y           [,1]      [,2]
    ##   1  0.000000000 0.0000000
    ##   2  0.000000000 0.0000000
    ##   3  0.000000000 0.0000000
    ##   4  0.000000000 0.0000000
    ##   5  0.022556391 0.1486708
    ##   6  0.021374046 0.1447383
    ##   7  0.037168142 0.1893413
    ##   8  0.007374631 0.0856216
    ##   9  0.023643950 0.1520430
    ##   10 0.043010753 0.2030634
    ##   11 0.027649770 0.1641565
    ##   12 0.040760870 0.1980049
    ##   13 0.032738095 0.1782157
    ##   14 0.047904192 0.2138840
    ##   15 0.038610039 0.1930367
    ##   16 0.061983471 0.2416253
    ##   17 0.000000000 0.0000000
    ##   18 0.000000000 0.0000000
    ##   19 0.000000000 0.0000000
    ##   20 0.000000000 0.0000000
    ##   21 0.000000000 0.0000000
    ## 
    ##     Sector_NoDur
    ## Y           [,1]       [,2]
    ##   1  0.000000000 0.00000000
    ##   2  0.000000000 0.00000000
    ##   3  0.041958042 0.20119803
    ##   4  0.056994819 0.23243567
    ##   5  0.100250627 0.30071095
    ##   6  0.106870229 0.30918430
    ##   7  0.104424779 0.30608179
    ##   8  0.057522124 0.23300946
    ##   9  0.095966620 0.29475050
    ##   10 0.093189964 0.29095929
    ##   11 0.073732719 0.26163703
    ##   12 0.059782609 0.23740641
    ##   13 0.032738095 0.17821571
    ##   14 0.032934132 0.17873197
    ##   15 0.034749035 0.18349808
    ##   16 0.033057851 0.17915821
    ##   17 0.009090909 0.09534626
    ##   18 0.000000000 0.00000000
    ##   19 0.000000000 0.00000000
    ##   20 0.000000000 0.00000000
    ##   21 0.000000000 0.00000000
    ## 
    ##     Sector_Other
    ## Y          [,1]      [,2]
    ##   1  0.04285714 0.2039973
    ##   2  0.12727273 0.3363500
    ##   3  0.11888112 0.3247862
    ##   4  0.15025907 0.3582545
    ##   5  0.05263158 0.2235772
    ##   6  0.12519084 0.3311881
    ##   7  0.11150442 0.3150347
    ##   8  0.16371681 0.3702916
    ##   9  0.15159944 0.3588819
    ##   10 0.14157706 0.3489286
    ##   11 0.15668203 0.3639202
    ##   12 0.22826087 0.4202830
    ##   13 0.27083333 0.4450530
    ##   14 0.19161677 0.3941636
    ##   15 0.15057915 0.3583306
    ##   16 0.17355372 0.3795102
    ##   17 0.20909091 0.4085206
    ##   18 0.56000000 0.5014265
    ##   19 0.33333333 0.5000000
    ##   20 0.42857143 0.5135526
    ##   21 0.71428571 0.4879500
    ## 
    ##     Sector_Shops
    ## Y          [,1]      [,2]
    ##   1  0.00000000 0.0000000
    ##   2  0.03636364 0.1889186
    ##   3  0.10489510 0.3074953
    ##   4  0.10362694 0.3055686
    ##   5  0.08771930 0.2832414
    ##   6  0.10992366 0.3130336
    ##   7  0.09734513 0.2966898
    ##   8  0.10914454 0.3120507
    ##   9  0.09874826 0.2985314
    ##   10 0.10215054 0.3031179
    ##   11 0.04608295 0.2099067
    ##   12 0.07608696 0.2654982
    ##   13 0.13690476 0.3442595
    ##   14 0.17664671 0.3819416
    ##   15 0.11969112 0.3252285
    ##   16 0.11983471 0.3254412
    ##   17 0.08181818 0.2753419
    ##   18 0.02000000 0.1414214
    ##   19 0.22222222 0.4409586
    ##   20 0.00000000 0.0000000
    ##   21 0.00000000 0.0000000
    ## 
    ##     Sector_Telcm
    ## Y          [,1]      [,2]
    ##   1  0.00000000 0.0000000
    ##   2  0.00000000 0.0000000
    ##   3  0.00000000 0.0000000
    ##   4  0.00000000 0.0000000
    ##   5  0.00000000 0.0000000
    ##   6  0.03053435 0.1721838
    ##   7  0.03008850 0.1709822
    ##   8  0.04277286 0.2024940
    ##   9  0.03198887 0.1760929
    ##   10 0.06810036 0.2521441
    ##   11 0.08986175 0.2863136
    ##   12 0.07608696 0.2654982
    ##   13 0.06250000 0.2424225
    ##   14 0.04191617 0.2006983
    ##   15 0.06949807 0.2547916
    ##   16 0.05371901 0.2259296
    ##   17 0.07272727 0.2608768
    ##   18 0.08000000 0.2740475
    ##   19 0.00000000 0.0000000
    ##   20 0.28571429 0.4688072
    ##   21 0.00000000 0.0000000
    ## 
    ##     Sector_Utils
    ## Y           [,1]       [,2]
    ##   1  0.071428571 0.25939889
    ##   2  0.000000000 0.00000000
    ##   3  0.034965035 0.18433693
    ##   4  0.176165803 0.38195197
    ##   5  0.127819549 0.33430797
    ##   6  0.160305344 0.36716939
    ##   7  0.230088496 0.42126224
    ##   8  0.185840708 0.38926525
    ##   9  0.126564673 0.33271613
    ##   10 0.082437276 0.27527654
    ##   11 0.080645161 0.27260364
    ##   12 0.040760870 0.19800495
    ##   13 0.014880952 0.12125704
    ##   14 0.026946108 0.16216891
    ##   15 0.007722008 0.08770449
    ##   16 0.066115702 0.24899923
    ##   17 0.009090909 0.09534626
    ##   18 0.020000000 0.14142136
    ##   19 0.000000000 0.00000000
    ##   20 0.071428571 0.26726124
    ##   21 0.142857143 0.37796447
    ## 
    ##     Current.Ratio
    ## Y        [,1]      [,2]
    ##   1  1.724241 0.6598883
    ##   2  1.540615 0.6311450
    ##   3  1.902789 0.9780043
    ##   4  1.851216 1.0055710
    ##   5  1.858375 0.9890133
    ##   6  1.530244 0.9441472
    ##   7  1.591007 1.5318330
    ##   8  1.602622 1.2234405
    ##   9  1.749943 0.9996589
    ##   10 1.860720 1.2424266
    ##   11 2.030566 1.6706550
    ##   12 2.288507 2.0364470
    ##   13 2.193354 1.6653989
    ##   14 2.124122 1.7480971
    ##   15 2.122145 2.3990766
    ##   16 2.232424 2.8020011
    ##   17 2.927445 4.2874105
    ##   18 5.312504 7.3441557
    ##   19 1.624878 1.0114224
    ##   20 3.971614 6.6707164
    ##   21 3.200857 3.1907775
    ## 
    ##     Long.term.Debt...Capital
    ## Y          [,1]       [,2]
    ##   1   0.1601543 0.10973061
    ##   2   0.1760018 0.09689222
    ##   3   0.3037350 0.17827574
    ##   4   0.3057964 0.17112383
    ##   5   0.3491045 0.19948149
    ##   6   0.3894211 0.19120684
    ##   7   0.3972296 0.15831052
    ##   8   0.4501555 0.33809768
    ##   9   0.4131257 0.19718958
    ##   10  0.4359934 0.25473191
    ##   11  0.4605111 0.29176381
    ##   12  0.5006951 0.28512807
    ##   13  0.5109786 0.24176821
    ##   14  0.5784105 0.40557129
    ##   15  0.8218193 1.22930434
    ##   16 -0.1455442 9.86727406
    ##   17  0.7233182 0.49120919
    ##   18  0.7270000 0.49925951
    ##   19  0.9548111 0.57680271
    ##   20  0.8825357 0.96444249
    ##   21  0.7316571 0.50827917
    ## 
    ##     Debt.Equity.Ratio
    ## Y           [,1]        [,2]
    ##   1    0.2784314   0.1923634
    ##   2    0.2743655   0.1598347
    ##   3    2.0129091  16.2202981
    ##   4    0.6226130   0.5376258
    ##   5    1.9456855  11.4223146
    ##   6   -1.3513922  55.2155309
    ##   7    0.8257619   1.2229847
    ##   8    0.6178487   4.6741839
    ##   9    0.1096391  29.9169203
    ##   10   0.2016057  35.0407667
    ##   11   0.1041859  38.1188884
    ##   12   1.8700310   6.4558121
    ##   13   0.4349509  37.8807333
    ##   14   1.1101569  26.6584863
    ##   15  -2.7186641  45.0628171
    ##   16   1.3638438   7.1337352
    ##   17   0.9084227   7.0890988
    ##   18 -27.5655400 208.6708792
    ##   19  -0.1006333   4.1174292
    ##   20  -0.1607286   3.1834779
    ##   21   1.4788429   2.9906743
    ## 
    ##     Gross.Margin
    ## Y        [,1]     [,2]
    ##   1  53.88239 19.81716
    ##   2  38.66151 18.41293
    ##   3  45.67766 20.06750
    ##   4  52.54762 23.34247
    ##   5  50.64574 23.11841
    ##   6  47.67629 22.06543
    ##   7  48.98291 24.10495
    ##   8  45.69345 24.14646
    ##   9  39.68298 21.78344
    ##   10 38.37337 22.45816
    ##   11 40.22137 22.40730
    ##   12 36.04664 23.21629
    ##   13 34.35827 21.12582
    ##   14 36.19326 24.17061
    ##   15 38.62414 27.06448
    ##   16 41.55122 27.13467
    ##   17 41.85816 22.70852
    ##   18 28.90431 21.11151
    ##   19 45.15674 26.25616
    ##   20 34.68396 27.28310
    ##   21 23.11134 11.83909
    ## 
    ##     Net.Profit.Margin
    ## Y           [,1]       [,2]
    ##   1   15.8552071   7.752961
    ##   2   13.1325400   6.929800
    ##   3   12.6423643   7.622626
    ##   4   12.5887984   8.787010
    ##   5   12.6280118   7.777639
    ##   6   10.2242214  11.884330
    ##   7   10.0100283  17.835857
    ##   8    9.5290608  10.616556
    ##   9    8.3134089  10.472421
    ##   10   5.4401839  15.169697
    ##   11   6.5784753  22.505252
    ##   12   5.2111242  13.385557
    ##   13   0.9284515  30.958430
    ##   14  -2.1685138  29.988871
    ##   15  -6.0766749  47.726855
    ##   16 -10.4409343  53.603293
    ##   17  -8.0437191  36.972240
    ##   18 -16.8446320  45.805300
    ##   19 -66.9699000 135.791348
    ##   20 -17.0697571  41.093290
    ##   21  -3.9751857   3.066246
    ## 
    ##     Asset.Turnover
    ## Y         [,1]      [,2]
    ##   1  0.6781814 0.3340922
    ##   2  0.9517400 0.3238886
    ##   3  0.9737790 0.7649848
    ##   4  0.9340135 1.0062929
    ##   5  0.9188837 0.7814736
    ##   6  0.8676260 0.8004508
    ##   7  0.8031927 0.7977809
    ##   8  0.7918811 0.6082304
    ##   9  0.8384983 0.6350479
    ##   10 0.8577283 0.6141863
    ##   11 0.7320650 0.5109361
    ##   12 0.8686563 0.7075293
    ##   13 0.9274012 0.6480699
    ##   14 0.9712769 0.8562309
    ##   15 0.9544695 0.6829309
    ##   16 0.7476231 0.5620558
    ##   17 0.6496845 0.4258539
    ##   18 0.7773120 0.4999941
    ##   19 0.7375444 0.6314315
    ##   20 0.7702929 0.3889448
    ##   21 0.8059714 0.5369512
    ## 
    ##     ROE...Return.On.Equity
    ## Y           [,1]       [,2]
    ##   1    20.244137   7.191553
    ##   2    21.630131   9.006998
    ##   3    42.719219 221.918210
    ##   4    17.912491  12.640034
    ##   5    49.322194 255.763820
    ##   6     3.280640 451.280921
    ##   7    17.161636  23.896671
    ##   8    13.790251  42.319596
    ##   9    26.942433 267.051826
    ##   10   12.512590 113.447867
    ##   11   16.732123  76.273374
    ##   12   18.662635  72.834008
    ##   13   -8.107919 248.122049
    ##   14    7.949318  99.873291
    ##   15   15.358022 254.672577
    ##   16  -15.944624 174.026157
    ##   17    5.689830 237.707715
    ##   18 -151.624366 818.410366
    ##   19  311.161389 737.625148
    ##   20   31.126229 108.240053
    ##   21   -5.325900  11.352393
    ## 
    ##     Return.On.Tangible.Equity
    ## Y           [,1]       [,2]
    ##   1   55.6096486   99.96487
    ##   2  392.0790527 1860.79311
    ##   3   53.9042126  219.84233
    ##   4   28.9314202  183.23430
    ##   5   74.3884752  707.20612
    ##   6   29.5368821  435.83822
    ##   7    3.4280283  100.06733
    ##   8   44.8608776  559.11620
    ##   9   29.1840715  203.50918
    ##   10  -0.9531559  606.86178
    ##   11   2.2972111  144.52882
    ##   12  -7.8269457  340.59792
    ##   13   0.1778664  525.24782
    ##   14  -9.1685138  136.47481
    ##   15  10.9643301  379.62317
    ##   16 -10.4617157  180.40456
    ##   17   7.5531245  114.86788
    ##   18   7.5277340  170.15681
    ##   19   1.4202667  307.38569
    ##   20  35.4088571  108.38729
    ##   21   4.6912143   18.14053
    ## 
    ##     ROA...Return.On.Assets
    ## Y          [,1]      [,2]
    ##   1    9.626977  4.287943
    ##   2   10.877582  3.981856
    ##   3    9.061198  4.059157
    ##   4    7.672409  4.791299
    ##   5    8.204552  4.481924
    ##   6    6.437119  5.597859
    ##   7    5.974575  5.612158
    ##   8    6.100624  5.026703
    ##   9    5.501802  4.905141
    ##   10   4.717031  6.235286
    ##   11   4.098590  8.803332
    ##   12   3.941093  7.978492
    ##   13   2.975519  9.097147
    ##   14   1.190627  8.917659
    ##   15  -1.608530 16.223179
    ##   16  -2.444130 17.111583
    ##   17  -2.093304 13.671776
    ##   18  -6.132044 17.997467
    ##   19 -21.877944 37.208022
    ##   20  -3.912050 16.186650
    ##   21  -1.904643  3.054156
    ## 
    ##     ROI...Return.On.Investment
    ## Y           [,1]      [,2]
    ##   1   17.1409743  6.535439
    ##   2   17.7004200  6.856915
    ##   3   14.6058769  6.964650
    ##   4   12.0575995  7.271008
    ##   5   13.3316439  8.283959
    ##   6   10.3976623  8.609637
    ##   7    9.9737108  8.399670
    ##   8   10.9449053 16.981206
    ##   9    8.6388153  7.940043
    ##   10   7.6030595 11.220544
    ##   11   5.5293634 13.108479
    ##   12   5.7013364 10.605181
    ##   13   4.5726616 12.494954
    ##   14   2.6043138 13.597681
    ##   15   0.1249039 26.440827
    ##   16 -16.5409029 98.519633
    ##   17  -3.1785900 20.680931
    ##   18  -8.4005520 25.685341
    ##   19 -33.2142111 48.358899
    ##   20 -11.5223857 33.968913
    ##   21  -2.5158429  5.852795
    ## 
    ##     Operating.Cash.Flow.Per.Share
    ## Y           [,1]      [,2]
    ##   1    0.2648957  1.573981
    ##   2    0.6979164  1.662937
    ##   3    0.5217811  1.772760
    ##   4    0.5397342  1.498401
    ##   5    0.3540855  1.328547
    ##   6    0.4316137  3.219860
    ##   7    0.4208267  2.402940
    ##   8    0.5810646  2.333217
    ##   9    0.9135078 12.547703
    ##   10   0.6273217  5.281231
    ##   11   0.5615454  4.964666
    ##   12   0.7326484  5.291282
    ##   13   0.5975762  4.010778
    ##   14  -0.1254389  3.533598
    ##   15   0.6939861  8.080053
    ##   16   0.1647570  9.067910
    ##   17  -0.7055227 11.765773
    ##   18   1.3878300 11.049660
    ##   19   3.2882667 37.797073
    ##   20 -11.1951571 27.886118
    ##   21   9.8031571 18.521164
    ## 
    ##     Free.Cash.Flow.Per.Share
    ## Y           [,1]      [,2]
    ##   1  -0.08142143  1.677302
    ##   2   0.18472909  2.203440
    ##   3   0.38748741  2.402222
    ##   4   0.39701503  1.881012
    ##   5   0.22829975  1.786921
    ##   6   0.09131450  5.733361
    ##   7   0.22317823  2.397430
    ##   8   0.35336947  4.872413
    ##   9   0.13356537 12.390486
    ##   10 -0.17135842  8.605481
    ##   11  0.44009908  3.198097
    ##   12  0.12828152  3.968288
    ##   13  0.13288095  3.072737
    ##   14 -0.30491976  3.606835
    ##   15  0.63421158 11.188980
    ##   16 -0.40290372 11.904445
    ##   17  0.38528818  8.980233
    ##   18  0.79036800 10.250533
    ##   19  0.42705556 27.796508
    ##   20 -9.52263571 22.801619
    ##   21  6.52482857 14.985467

## Model Validation

``` r
# Predict the samples from test data using the model
result <- predict(nb.model, test)

# Print the Confusion matrix
confusion.matrix <- confusionMatrix(as.factor(result), as.factor(test$Rating))
plot.custom.confusion.matrix(confusion.matrix$table)
```

![](class_naive_bayes_files/figure-gfm/unnamed-chunk-4-1.png)<!-- -->

``` r
# Print the accuracy stats of the model
kable(data.frame(confusion.matrix$overall))
```

|                | confusion.matrix.overall |
|:---------------|-------------------------:|
| Accuracy       |                0.0245637 |
| Kappa          |                0.0176265 |
| AccuracyLower  |                0.0174400 |
| AccuracyUpper  |                0.0335608 |
| AccuracyNull   |                0.1215255 |
| AccuracyPValue |                1.0000000 |
| McnemarPValue  |                      NaN |

``` r
# Print validation stats of the model
kable(data.frame(confusion.matrix$byClass))
```

|           | Sensitivity | Specificity | Pos.Pred.Value | Neg.Pred.Value | Precision |    Recall |        F1 | Prevalence | Detection.Rate | Detection.Prevalence | Balanced.Accuracy |
|:----------|------------:|------------:|---------------:|---------------:|----------:|----------:|----------:|-----------:|---------------:|---------------------:|------------------:|
| Class: 1  |   0.7222222 |   0.8606933 |      0.0575221 |      0.9962150 | 0.0575221 | 0.7222222 | 0.1065574 |  0.0116354 |      0.0084034 |            0.1460892 |         0.7914577 |
| Class: 2  |   0.8000000 |   0.7160574 |      0.0268456 |      0.9972727 | 0.0268456 | 0.8000000 | 0.0519481 |  0.0096962 |      0.0077569 |            0.2889463 |         0.7580287 |
| Class: 3  |   0.0000000 |   0.9920266 |      0.0000000 |      0.9726384 | 0.0000000 | 0.0000000 |       NaN |  0.0271493 |      0.0000000 |            0.0077569 |         0.4960133 |
| Class: 4  |   0.1346154 |   0.9638796 |      0.1147541 |      0.9697174 | 0.1147541 | 0.1346154 | 0.1238938 |  0.0336134 |      0.0045249 |            0.0394312 |         0.5492475 |
| Class: 5  |   0.0000000 |   1.0000000 |            NaN |      0.9495798 |        NA | 0.0000000 |        NA |  0.0504202 |      0.0000000 |            0.0000000 |         0.5000000 |
| Class: 6  |   0.0000000 |   1.0000000 |            NaN |      0.8907563 |        NA | 0.0000000 |        NA |  0.1092437 |      0.0000000 |            0.0000000 |         0.5000000 |
| Class: 7  |   0.0000000 |   1.0000000 |            NaN |      0.9062702 |        NA | 0.0000000 |        NA |  0.0937298 |      0.0000000 |            0.0000000 |         0.5000000 |
| Class: 8  |   0.0000000 |   1.0000000 |            NaN |      0.8959276 |        NA | 0.0000000 |        NA |  0.1040724 |      0.0000000 |            0.0000000 |         0.5000000 |
| Class: 9  |   0.0000000 |   1.0000000 |            NaN |      0.8784745 |        NA | 0.0000000 |        NA |  0.1215255 |      0.0000000 |            0.0000000 |         0.5000000 |
| Class: 10 |   0.0000000 |   1.0000000 |            NaN |      0.9069166 |        NA | 0.0000000 |        NA |  0.0930834 |      0.0000000 |            0.0000000 |         0.5000000 |
| Class: 11 |   0.0000000 |   1.0000000 |            NaN |      0.9288946 |        NA | 0.0000000 |        NA |  0.0711054 |      0.0000000 |            0.0000000 |         0.5000000 |
| Class: 12 |   0.0000000 |   1.0000000 |            NaN |      0.9450549 |        NA | 0.0000000 |        NA |  0.0549451 |      0.0000000 |            0.0000000 |         0.5000000 |
| Class: 13 |   0.0000000 |   1.0000000 |            NaN |      0.9411765 |        NA | 0.0000000 |        NA |  0.0588235 |      0.0000000 |            0.0000000 |         0.5000000 |
| Class: 14 |   0.0113636 |   1.0000000 |      1.0000000 |      0.9437257 | 1.0000000 | 0.0113636 | 0.0224719 |  0.0568843 |      0.0006464 |            0.0006464 |         0.5056818 |
| Class: 15 |   0.0196078 |   0.9993316 |      0.5000000 |      0.9676375 | 0.5000000 | 0.0196078 | 0.0377358 |  0.0329670 |      0.0006464 |            0.0012928 |         0.5094697 |
| Class: 16 |   0.0000000 |   0.9993338 |      0.0000000 |      0.9702458 | 0.0000000 | 0.0000000 |       NaN |  0.0297350 |      0.0000000 |            0.0006464 |         0.4996669 |
| Class: 17 |   0.0000000 |   1.0000000 |            NaN |      0.9741435 |        NA | 0.0000000 |        NA |  0.0258565 |      0.0000000 |            0.0000000 |         0.5000000 |
| Class: 18 |   0.0000000 |   0.9954308 |      0.0000000 |      0.9902597 | 0.0000000 | 0.0000000 |       NaN |  0.0096962 |      0.0000000 |            0.0045249 |         0.4977154 |
| Class: 19 |   1.0000000 |   0.7742561 |      0.0028571 |      1.0000000 | 0.0028571 | 1.0000000 | 0.0056980 |  0.0006464 |      0.0006464 |            0.2262443 |         0.8871281 |
| Class: 20 |   0.2500000 |   0.9416721 |      0.0109890 |      0.9979396 | 0.0109890 | 0.2500000 | 0.0210526 |  0.0025856 |      0.0006464 |            0.0588235 |         0.5958360 |
| Class: 21 |   0.5000000 |   0.7751134 |      0.0057307 |      0.9983306 | 0.0057307 | 0.5000000 | 0.0113314 |  0.0025856 |      0.0012928 |            0.2255979 |         0.6375567 |

``` r
# Save the results
algorithm <- "Naive.Bayes"
save.class.acc.result(confusion.matrix$overall, algorithm)
save.class.pvv.result(confusion.matrix$byClass, algorithm)
```
