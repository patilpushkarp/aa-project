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
```

## Model Training

The model is first trained on the training data and then evaluated on
testing data.

``` r
# Model training
multinom.model <- multinom(Rating~., data=train, )
```

    ## # weights:  693 (640 variable)
    ## initial  value 18869.950069 
    ## iter  10 value 17471.805981
    ## iter  20 value 17404.039493
    ## iter  30 value 17033.372536
    ## iter  40 value 16921.451633
    ## iter  50 value 16779.168679
    ## iter  60 value 16699.983458
    ## iter  70 value 16571.429673
    ## iter  80 value 16032.803652
    ## iter  90 value 15310.464460
    ## iter 100 value 14537.712345
    ## final  value 14537.712345 
    ## stopped after 100 iterations

``` r
summary(multinom.model)
```

    ## Call:
    ## multinom(formula = Rating ~ ., data = train)
    ## 
    ## Coefficients:
    ##     (Intercept)           X.1             X
    ## 2  -0.156996443  4.958836e-05  4.958836e-05
    ## 3  -0.319174135  4.682236e-05  4.682236e-05
    ## 4  -0.360076895  5.523652e-05  5.523652e-05
    ## 5  -0.446814676  1.521011e-05  1.521010e-05
    ## 6   0.006007988  9.417033e-05  9.417033e-05
    ## 7  -0.028561034  4.123743e-05  4.123743e-05
    ## 8   0.304496775  3.236887e-05  3.236887e-05
    ## 9   0.532781417 -6.881455e-06 -6.881454e-06
    ## 10  0.289446446  2.239731e-05  2.239731e-05
    ## 11  0.102628628  6.298840e-05  6.298840e-05
    ## 12  0.158091908  1.765425e-07  1.765459e-07
    ## 13  0.191779684  2.256009e-05  2.256009e-05
    ## 14  0.190702333 -4.510291e-05 -4.510291e-05
    ## 15  0.143328431 -4.526875e-05 -4.526875e-05
    ## 16 -0.019437318 -2.778290e-06 -2.778290e-06
    ## 17 -0.117889176 -9.766223e-06 -9.766224e-06
    ## 18 -0.018941470 -2.348232e-04 -2.348232e-04
    ## 19 -0.062235996 -5.153366e-04 -5.153366e-04
    ## 20 -0.095416445 -1.538041e-04 -1.538041e-04
    ## 21 -0.005758271 -8.748526e-04 -8.748526e-04
    ##    Rating.Agency_Egan.Jones.Ratings.Company Rating.Agency_Fitch.Ratings
    ## 2                               0.118691839               -0.0410671732
    ## 3                              -0.040911166               -0.0882920441
    ## 4                               0.527650899               -0.0790962237
    ## 5                               1.038350716               -0.1592321065
    ## 6                               0.281046635                0.0009883002
    ## 7                               0.712852626               -0.1405433651
    ## 8                              -0.038966398                0.2812289940
    ## 9                              -0.001247364                0.1092573869
    ## 10                             -0.408703487                0.3150433988
    ## 11                              0.088297181                0.0370507214
    ## 12                             -0.416994626                0.0194631364
    ## 13                             -0.473797337               -0.0658604375
    ## 14                             -0.357550593               -0.0856222012
    ## 15                             -0.246728732               -0.0085720096
    ## 16                             -0.321440524               -0.0325746827
    ## 17                             -0.180769033               -0.0066965663
    ## 18                             -0.006413146                0.0292778081
    ## 19                             -0.014813905               -0.0188143883
    ## 20                             -0.033198919               -0.0037278873
    ## 21                              0.019350921                0.0058218963
    ##    Rating.Agency_Moody.s.Investors.Service
    ## 2                             -0.044756793
    ## 3                             -0.251020574
    ## 4                             -0.363695535
    ## 5                             -0.416423266
    ## 6                             -0.656941305
    ## 7                             -0.060020300
    ## 8                              0.227291906
    ## 9                              0.217646678
    ## 10                             0.159983510
    ## 11                            -0.103405374
    ## 12                             0.297692139
    ## 13                             0.263797104
    ## 14                             0.328655701
    ## 15                             0.193018731
    ## 16                             0.293052907
    ## 17                             0.119117117
    ## 18                            -0.015875690
    ## 19                             0.003339274
    ## 20                            -0.007384472
    ## 21                            -0.014792644
    ##    Rating.Agency_Standard...Poor.s.Ratings.Services Binary.Rating  Sector_BusEq
    ## 2                                       -0.18986432    0.09037909  0.0332960830
    ## 3                                        0.06104965    0.16892973  0.0884796814
    ## 4                                       -0.44493603    0.28869326  0.0154916505
    ## 5                                       -0.90951002    0.73837173  0.0100379763
    ## 6                                        0.38091436    1.65040056 -0.0948900746
    ## 7                                       -0.54085000    1.42591754 -0.0639644466
    ## 8                                       -0.16505773    1.88126011 -0.1987657888
    ## 9                                        0.20712472    2.28982418  0.0275417500
    ## 10                                       0.22312302    1.93095227  0.2342326522
    ## 11                                       0.08068610   -2.34490679  0.0513150381
    ## 12                                       0.25793126   -1.89473594  0.0747815620
    ## 13                                       0.46764036   -1.62548615  0.0167235193
    ## 14                                       0.30521943   -1.55500888 -0.0531719140
    ## 15                                       0.20561044   -1.09210525 -0.0005923543
    ## 16                                       0.04152498   -1.08512518 -0.0737437606
    ## 17                                      -0.04954069   -0.52657295 -0.0869884869
    ## 18                                      -0.02593044   -0.19643795  0.0016129910
    ## 19                                      -0.03194698   -0.06491431 -0.0202871468
    ## 20                                      -0.05110517   -0.09157478 -0.0201613706
    ## 21                                      -0.01613844   -0.03095713 -0.0049774508
    ##    Sector_Chems Sector_Durbl  Sector_Enrgy   Sector_Hlth  Sector_Manuf
    ## 2  -0.041004120 -0.036555443  0.1768225180  0.0772916261 -0.1063945092
    ## 3  -0.058165535 -0.065205830  0.0458561074  0.0390282777  0.0417876801
    ## 4  -0.087465692 -0.016328278 -0.1374744684  0.2156379906 -0.1232415551
    ## 5  -0.009773438  0.074777441 -0.1278630885  0.2296910237  0.0003917602
    ## 6   0.113955026  0.007626721  0.0529015963 -0.2804164950 -0.0172511691
    ## 7  -0.161004967 -0.068486320 -0.0627742403 -0.0459099561 -0.0736518369
    ## 8  -0.051839072 -0.013925839  0.1933937609 -0.0936500049  0.0283751726
    ## 9   0.318137792 -0.048725395 -0.0721177187 -0.2687266879  0.1619480656
    ## 10  0.122377315 -0.029496895 -0.1856235999 -0.1508295540  0.0263733630
    ## 11 -0.022032448  0.005536622 -0.0041128110 -0.0421904015  0.1917241651
    ## 12 -0.100818484  0.045439889 -0.0363747579 -0.0914694894  0.1385762783
    ## 13 -0.003378777 -0.033698632 -0.0963250162 -0.0700749987  0.1609482876
    ## 14  0.157756576  0.003537085 -0.0662109935  0.0529239973 -0.0117957049
    ## 15 -0.029715671  0.179893326  0.1403156593  0.0600256859 -0.0841898412
    ## 16 -0.030564257  0.078010280 -0.0013150369  0.0682119662 -0.0907693150
    ## 17 -0.036022763 -0.020131738  0.0917913894  0.1137408127  0.0055330273
    ## 18 -0.014158086 -0.010036978 -0.0288172414  0.0164264794 -0.0504757241
    ## 19 -0.008252312 -0.005511386  0.0293867072 -0.0034377297 -0.0203722009
    ## 20 -0.010690161 -0.008436918  0.0091761673 -0.0035888463 -0.0411668951
    ## 21 -0.003605097 -0.003270573 -0.0005379097 -0.0003624916 -0.0116309020
    ##    Sector_Money Sector_NoDur  Sector_Other  Sector_Shops  Sector_Telcm
    ## 2  -0.014820891  -0.04664762 -0.0462584603 -0.0861722010 -0.0314527944
    ## 3  -0.045628853  -0.08122560 -0.1033335527 -0.0527299008 -0.0577977012
    ## 4  -0.064976883  -0.06431645 -0.0546406483 -0.0184847915 -0.0990585940
    ## 5  -0.034145294   0.05482662 -0.4382001706 -0.0700209109 -0.2001899659
    ## 6  -0.051217044   0.18907318 -0.1189540727  0.0899348508 -0.1084224628
    ## 7   0.034456628   0.15487217 -0.1522515305  0.0495591346 -0.0956471708
    ## 8  -0.126379232  -0.07373806  0.1934172740  0.0812800715 -0.0075374818
    ## 9  -0.002127496   0.17047332  0.1101846997  0.0099559945 -0.0354869841
    ## 10  0.098680658   0.11733112 -0.0284963026 -0.0003703756  0.1519069888
    ## 11 -0.004701477   0.03426156 -0.0016838627 -0.1879051720  0.1885593129
    ## 12  0.058009427  -0.01691943  0.2023937560 -0.0775003928  0.1270787721
    ## 13  0.032490025  -0.09061104  0.3089459945  0.1002733228  0.0731094301
    ## 14  0.078024468  -0.07738175  0.0851427045  0.2142024269  0.0054139704
    ## 15  0.032365477  -0.03644266 -0.0125579161  0.0696196495  0.0560919313
    ## 16  0.086513706  -0.04694618  0.0085407706  0.0680224187  0.0024089459
    ## 17 -0.029691967  -0.05090368  0.0083041010 -0.0160896878  0.0223460291
    ## 18 -0.011256133  -0.02701649  0.1391469597 -0.0403737112  0.0245160221
    ## 19 -0.003127342  -0.02274844 -0.0005949641  0.0069157730 -0.0061086706
    ## 20 -0.005324730  -0.01199158 -0.0029011575 -0.0303254542  0.0331102118
    ## 21 -0.001247579  -0.00367617  0.0287893918 -0.0093853058 -0.0008229559
    ##    Sector_Utils Current.Ratio Long.term.Debt...Capital Debt.Equity.Ratio
    ## 2  -0.035100630   -0.47069756             -0.109903125      0.0201891352
    ## 3  -0.070238907   -0.55377233             -0.299959869      0.0140977675
    ## 4   0.074780827   -0.60516089             -0.408088447      0.0366701217
    ## 5   0.063653370   -0.79348915             -0.604081021      0.0052855124
    ## 6   0.223667928   -1.19991723             -0.344352027     -0.0003996838
    ## 7   0.456241496   -0.62487706             -0.351070165     -0.0072117517
    ## 8   0.373865970   -0.06285417             -0.135578613      0.0232085449
    ## 9   0.161724075    0.03159802              0.088234564     -0.0058493631
    ## 10 -0.066638922    0.27532821              0.133309903     -0.0060939832
    ## 11 -0.106141896    0.44095286              0.186651890     -0.0022123172
    ## 12 -0.165105223    0.61466112              0.287893400      0.0182144757
    ## 13 -0.206622433    0.59243533              0.283447389      0.0113877226
    ## 14 -0.197738532    0.61782898              0.359720975      0.0036236333
    ## 15 -0.231484859    0.63487537              0.656150101     -0.0097230552
    ## 16 -0.087806855    0.45689712              0.423332193      0.0146677939
    ## 17 -0.119776209    0.60350277              0.124411890      0.0189070587
    ## 18 -0.018509561    0.92315384             -0.005358539     -0.0071915195
    ## 19 -0.008098287   -0.25876166              0.060710573      0.0037309707
    ## 20 -0.003115715    0.01299274             -0.021040853     -0.0129242117
    ## 21  0.004968773    0.05077082              0.016150260      0.0173219978
    ##    Gross.Margin Net.Profit.Margin Asset.Turnover ROE...Return.On.Equity
    ## 2    0.04872663      -0.016334470    -0.27705976           0.0038520553
    ## 3    0.06648315      -0.090569581    -0.33449012          -0.0001533655
    ## 4    0.06718761      -0.092663801    -0.11418353          -0.0034989775
    ## 5    0.07256144      -0.072510238    -0.08545068           0.0040882145
    ## 6    0.06474003      -0.049444745     0.35320026           0.0030510796
    ## 7    0.06435338      -0.074211931     0.14377750           0.0047790631
    ## 8    0.04648560      -0.063436151     0.06647311          -0.0010935896
    ## 9    0.04222362      -0.052743209     0.20745676           0.0039361068
    ## 10   0.02769283      -0.058657661     0.07049660           0.0024342699
    ## 11   0.05427737      -0.072759192    -0.23479042           0.0041710067
    ## 12   0.04749250      -0.074618070     0.10880084           0.0019983990
    ## 13   0.04108658      -0.036984610     0.22856529           0.0012284274
    ## 14   0.05241757      -0.025398524     0.48499425           0.0023964320
    ## 15   0.04265795      -0.078410794     0.45687419           0.0049811134
    ## 16   0.05285119      -0.077036940    -0.08878048           0.0015802156
    ## 17   0.04066774      -0.080267434    -0.21805919           0.0004792161
    ## 18   0.02092524      -0.007113987    -0.10659352           0.0033376737
    ## 19   0.06003582      -0.069299271    -0.06452142           0.0034260868
    ## 20   0.05195647      -0.041409660    -0.14411168           0.0038859788
    ## 21   0.06100981      -0.012150019    -0.02571289          -0.0001900184
    ##    Return.On.Tangible.Equity ROA...Return.On.Assets ROI...Return.On.Investment
    ## 2              -3.526376e-04            -0.23881174                0.017005667
    ## 3              -2.691734e-04             0.17565453               -0.065261527
    ## 4              -5.826409e-04             0.15677762               -0.067701093
    ## 5               4.878850e-05             0.11409520               -0.044242629
    ## 6              -9.590491e-05             0.05987305               -0.076430368
    ## 7              -8.020162e-04             0.03086346               -0.057664782
    ## 8              -8.226647e-04            -0.09036805               -0.011496178
    ## 9              -2.805013e-04            -0.08483755               -0.064231972
    ## 10             -1.015154e-03            -0.18215023               -0.004728667
    ## 11             -1.482497e-03             0.16458446               -0.135107579
    ## 12             -4.460706e-04             0.06456525               -0.121137010
    ## 13             -8.227442e-04            -0.08953888               -0.055256252
    ## 14             -8.953523e-04            -0.15087285               -0.046502858
    ## 15             -1.016688e-03             0.09757622               -0.156077798
    ## 16             -1.713644e-03             0.14013611               -0.145801918
    ## 17             -3.366621e-04             0.10876890               -0.131666582
    ## 18              2.311464e-04            -0.29797741               -0.015594087
    ## 19             -1.975942e-03            -0.14426292               -0.077405863
    ## 20              2.528965e-04            -0.17342789               -0.064489566
    ## 21             -1.389813e-04            -0.09649686               -0.054230647
    ##    Operating.Cash.Flow.Per.Share Free.Cash.Flow.Per.Share
    ## 2                    0.021073297              0.007922329
    ## 3                   -0.009647424              0.021327067
    ## 4                    0.022696437              0.025665653
    ## 5                    0.057457633             -0.023840743
    ## 6                    0.043478452             -0.011054732
    ## 7                    0.067591757              0.012837973
    ## 8                    0.010433250              0.018980114
    ## 9                    0.060865985              0.001349415
    ## 10                   0.038435144             -0.019301332
    ## 11                   0.024132261              0.033502560
    ## 12                  -0.032609846              0.054018673
    ## 13                  -0.048088294              0.047904711
    ## 14                  -0.021397187             -0.023843372
    ## 15                   0.046017124              0.029306206
    ## 16                   0.014050478             -0.035423686
    ## 17                   0.071726057              0.012064345
    ## 18                   0.041532457              0.040502513
    ## 19                  -0.006119091             -0.031250031
    ## 20                   0.029334845              0.025824537
    ## 21                  -0.072790295             -0.047341769
    ## 
    ## Std. Errors:
    ##     (Intercept)          X.1            X
    ## 2  0.0004784116 3.095125e-05 3.095125e-05
    ## 3  0.0010786271 2.446379e-05 2.446379e-05
    ## 4  0.0005547492 2.466896e-05 2.466896e-05
    ## 5  0.0007657500 2.407301e-05 2.407301e-05
    ## 6  0.0034663861 2.182357e-05 2.182357e-05
    ## 7  0.0018932414 2.239701e-05 2.239701e-05
    ## 8  0.0078201414 2.196452e-05 2.196452e-05
    ## 9  0.0073141792 2.169886e-05 2.169886e-05
    ## 10 0.0061753611 2.248532e-05 2.248532e-05
    ## 11 0.0025069700 2.299329e-05 2.299329e-05
    ## 12 0.0019336539 2.332744e-05 2.332744e-05
    ## 13 0.0023149829 2.295672e-05 2.295672e-05
    ## 14 0.0027054204 2.311135e-05 2.311135e-05
    ## 15 0.0022378460 2.324070e-05 2.324070e-05
    ## 16 0.0035780586 2.404856e-05 2.404856e-05
    ## 17 0.0021241498 2.554945e-05 2.554945e-05
    ## 18 0.0015907226 4.042639e-05 4.042639e-05
    ## 19 0.0002107055 1.179975e-04 1.179975e-04
    ## 20 0.0001703795 4.740617e-05 4.740617e-05
    ## 21 0.0006284858 1.931949e-04 1.931949e-04
    ##    Rating.Agency_Egan.Jones.Ratings.Company Rating.Agency_Fitch.Ratings
    ## 2                              3.298760e-04                1.235724e-04
    ## 3                              4.016117e-04                1.455041e-04
    ## 4                              2.494167e-04                7.375535e-05
    ## 5                              5.288192e-04                9.528115e-05
    ## 6                              2.763610e-03                5.668239e-05
    ## 7                              1.744076e-03                1.040096e-04
    ## 8                              5.425207e-03                3.227224e-04
    ## 9                              3.842845e-03                2.546469e-04
    ## 10                             3.145419e-03                2.778722e-04
    ## 11                             1.614288e-03                9.031907e-05
    ## 12                             8.379811e-04                9.117493e-05
    ## 13                             7.052399e-04                8.719911e-05
    ## 14                             8.176339e-04                9.795674e-05
    ## 15                             9.494179e-04                1.027136e-04
    ## 16                             1.520630e-03                1.384269e-04
    ## 17                             1.355985e-03                8.765570e-05
    ## 18                             3.527155e-04                1.627045e-04
    ## 19                             3.786898e-04                5.948187e-05
    ## 20                             1.842809e-04                3.397185e-05
    ## 21                             6.537304e-05                5.975873e-05
    ##    Rating.Agency_Moody.s.Investors.Service
    ## 2                             3.701662e-04
    ## 3                             4.224132e-04
    ## 4                             2.483959e-04
    ## 5                             1.956996e-04
    ## 6                             3.806662e-04
    ## 7                             2.639546e-04
    ## 8                             3.457866e-04
    ## 9                             5.258097e-04
    ## 10                            6.943072e-04
    ## 11                            2.095052e-04
    ## 12                            5.529083e-04
    ## 13                            5.264875e-04
    ## 14                            7.324285e-04
    ## 15                            5.659854e-04
    ## 16                            7.769287e-04
    ## 17                            3.888060e-04
    ## 18                            9.548790e-04
    ## 19                            1.320152e-04
    ## 20                            8.269713e-05
    ## 21                            2.611117e-04
    ##    Rating.Agency_Standard...Poor.s.Ratings.Services Binary.Rating Sector_BusEq
    ## 2                                      0.0002514329  0.0008803419 0.0003216813
    ## 3                                      0.0004718422  0.0016578408 0.0008001357
    ## 4                                      0.0002884393  0.0005748954 0.0003399576
    ## 5                                      0.0002173111  0.0010733781 0.0003619565
    ## 6                                      0.0011262328  0.0029305322 0.0005156625
    ## 7                                      0.0003763113  0.0014638270 0.0008899250
    ## 8                                      0.0019180007  0.0066763390 0.0018232429
    ## 9                                      0.0028334641  0.0062128693 0.0012701668
    ## 10                                     0.0021924811  0.0047590442 0.0016621715
    ## 11                                     0.0008516472  0.0006190450 0.0013399690
    ## 12                                     0.0008986488  0.0007715690 0.0007342335
    ## 13                                     0.0010793082  0.0008407753 0.0006104383
    ## 14                                     0.0012394989  0.0010465429 0.0008249959
    ## 15                                     0.0008561720  0.0013740975 0.0006850506
    ## 16                                     0.0011757970  0.0012245387 0.0013736274
    ## 17                                     0.0004357053  0.0010075699 0.0008300774
    ## 18                                     0.0009518876  0.0006168995 0.0005661593
    ## 19                                     0.0002465116  0.0001222147 0.0000527170
    ## 20                                     0.0001280917  0.0002773241 0.0001096184
    ## 21                                     0.0004229254  0.0003422509 0.0001423270
    ##    Sector_Chems Sector_Durbl Sector_Enrgy  Sector_Hlth Sector_Manuf
    ## 2  1.012101e-04 1.436024e-04 0.0006016802 1.502878e-04 3.608288e-04
    ## 3  1.562295e-04 8.175378e-05 0.0004890383 5.786351e-04 3.927122e-04
    ## 4  7.013612e-05 8.487452e-05 0.0003290721 2.998370e-04 1.831853e-04
    ## 5  1.068294e-04 1.033083e-04 0.0003131181 3.727284e-04 3.568110e-04
    ## 6  1.987895e-04 1.852533e-04 0.0004615597 3.320421e-04 6.908784e-04
    ## 7  1.628902e-04 1.227743e-04 0.0002247257 7.004445e-04 5.851836e-04
    ## 8  8.043487e-04 5.381421e-04 0.0002604331 3.002466e-04 3.072051e-03
    ## 9  7.811084e-04 3.342212e-04 0.0002608857 2.384691e-04 2.299303e-03
    ## 10 8.134753e-04 3.789527e-04 0.0003650383 1.451088e-04 2.371820e-03
    ## 11 1.778811e-04 9.492618e-05 0.0006759084 3.084672e-04 1.387169e-03
    ## 12 1.734878e-04 2.181353e-04 0.0004120428 3.845177e-04 7.496305e-04
    ## 13 2.048146e-04 5.111620e-04 0.0003418516 2.692153e-04 7.545163e-04
    ## 14 2.875989e-04 2.857811e-04 0.0004675984 2.819234e-04 8.259714e-04
    ## 15 3.838221e-04 1.347771e-04 0.0004335330 3.227814e-04 6.423809e-04
    ## 16 3.659662e-04 1.381218e-04 0.0006825676 3.721161e-04 1.400720e-03
    ## 17 1.349853e-04 8.598131e-05 0.0005004421 2.880744e-04 5.120922e-04
    ## 18 1.191067e-04 6.317957e-05 0.0002068098 2.226032e-04 1.477616e-04
    ## 19 7.662048e-05 4.347994e-05 0.0002316107 3.963867e-05 1.530857e-05
    ## 20 9.309969e-05 4.577302e-05 0.0001786462 3.603588e-05 4.278378e-05
    ## 21 3.481792e-05 8.785806e-05 0.0001124115 2.398481e-04 2.839188e-04
    ##    Sector_Money Sector_NoDur Sector_Other Sector_Shops Sector_Telcm
    ## 2  2.325341e-04 2.341840e-04 0.0001362289 2.134867e-04 1.123198e-04
    ## 3  1.073504e-04 2.405839e-04 0.0003029057 3.426771e-04 4.381951e-04
    ## 4  2.556556e-05 1.708401e-04 0.0001241412 2.715786e-04 9.010241e-05
    ## 5  7.968103e-05 3.445473e-04 0.0001330143 3.554792e-04 1.899896e-04
    ## 6  5.207509e-05 4.306209e-04 0.0001767081 1.940291e-03 2.792348e-04
    ## 7  4.152536e-05 8.718099e-04 0.0001300882 6.907598e-04 1.379936e-04
    ## 8  7.916901e-05 3.611504e-04 0.0009333174 2.071617e-03 7.152400e-04
    ## 9  2.719746e-04 4.526443e-04 0.0008174208 3.335182e-03 3.954587e-04
    ## 10 1.344104e-04 3.426698e-04 0.0018977062 1.153105e-03 4.431103e-04
    ## 11 1.705247e-04 1.688986e-04 0.0013303813 2.477816e-04 4.238906e-04
    ## 12 1.200581e-04 9.113101e-05 0.0015423412 1.562682e-04 2.330410e-04
    ## 13 1.264741e-04 1.215049e-04 0.0017345199 1.903810e-04 2.213734e-04
    ## 14 1.127820e-04 1.522667e-04 0.0015255493 4.847198e-04 2.443106e-04
    ## 15 1.017291e-04 1.148021e-04 0.0016350069 3.018755e-04 2.015090e-04
    ## 16 1.237105e-04 1.564999e-04 0.0018909559 3.736481e-04 4.226356e-04
    ## 17 4.921418e-05 8.831710e-05 0.0021444021 9.119364e-05 2.490400e-04
    ## 18 2.409484e-04 8.234584e-05 0.0022023369 1.707289e-04 1.473579e-04
    ## 19 2.069869e-05 6.791975e-05 0.0003609300 6.724551e-05 8.791115e-05
    ## 20 2.701540e-05 4.528108e-05 0.0002327662 9.940486e-05 7.343464e-05
    ## 21 6.792462e-05 4.153769e-05 0.0003586659 1.881108e-04 8.676092e-05
    ##    Sector_Utils Current.Ratio Long.term.Debt...Capital Debt.Equity.Ratio
    ## 2  3.372863e-04  0.0010872082             0.0007247510       0.012388049
    ## 3  6.055640e-04  0.0064901797             0.0007895725       0.015867179
    ## 4  3.957733e-04  0.0024740447             0.0027411690       0.008951779
    ## 5  6.100238e-04  0.0031879907             0.0029629885       0.015788445
    ## 6  9.213979e-04  0.0082094785             0.0026107855       0.012661390
    ## 7  1.635824e-03  0.0087874416             0.0012132332       0.006732434
    ## 8  1.241743e-03  0.0309805508             0.0020821988       0.009258523
    ## 9  1.293704e-03  0.0240131667             0.0024267132       0.006764432
    ## 10 1.132349e-03  0.0327506639             0.0046545965       0.006644440
    ## 11 3.480420e-04  0.0323358409             0.0032186607       0.009672479
    ## 12 2.256726e-04  0.0254765508             0.0010479103       0.009992739
    ## 13 2.641019e-04  0.0259691572             0.0046693060       0.010807955
    ## 14 2.777987e-04  0.0253872570             0.0046274618       0.010536396
    ## 15 1.949634e-04  0.0254034191             0.0009158934       0.006586325
    ## 16 4.683779e-04  0.0357653294             0.0010199775       0.011985148
    ## 17 3.386049e-04  0.0348913797             0.0035251424       0.013289530
    ## 18 1.912796e-04  0.0313427247             0.0010468691       0.011368038
    ## 19 5.316420e-05  0.0008973702             0.0004464841       0.036172142
    ## 20 3.242528e-05  0.0005470705             0.0001743688       0.006488736
    ## 21 2.642506e-04  0.0010599193             0.0003388440       0.032635896
    ##    Gross.Margin Net.Profit.Margin Asset.Turnover ROE...Return.On.Equity
    ## 2   0.007122333       0.013762097   0.0016154536            0.001559986
    ## 3   0.005746433       0.008948320   0.0021312879            0.002049277
    ## 4   0.005968164       0.008295666   0.0013411213            0.001511501
    ## 5   0.005823046       0.010354044   0.0020217081            0.001625882
    ## 6   0.005703840       0.008842558   0.0097277431            0.001708077
    ## 7   0.005712748       0.008667085   0.0038850958            0.001509865
    ## 8   0.005661775       0.007781427   0.0136228681            0.001511427
    ## 9   0.005649632       0.007684416   0.0170113299            0.001532941
    ## 10  0.005815028       0.007560445   0.0091413123            0.001581169
    ## 11  0.005893196       0.008674004   0.0016310666            0.001662399
    ## 12  0.005954457       0.007465687   0.0014597935            0.001590823
    ## 13  0.006004618       0.009291050   0.0019480347            0.001755151
    ## 14  0.005859098       0.008894479   0.0034973728            0.001743670
    ## 15  0.005930625       0.007249814   0.0025089981            0.001539006
    ## 16  0.005991313       0.008136402   0.0030447341            0.002061776
    ## 17  0.006262956       0.007793903   0.0012120705            0.002384360
    ## 18  0.007702145       0.011478142   0.0015161091            0.001927332
    ## 19  0.009228688       0.009378991   0.0001363178            0.001604322
    ## 20  0.007674861       0.011832710   0.0002410101            0.001556482
    ## 21  0.008904156       0.024347052   0.0014778147            0.005112738
    ##    Return.On.Tangible.Equity ROA...Return.On.Assets ROI...Return.On.Investment
    ## 2               0.0006650551            0.031656072                0.006366596
    ## 3               0.0003135561            0.027588002                0.017276762
    ## 4               0.0004779501            0.017436564                0.008760636
    ## 5               0.0001289000            0.018153605                0.009612211
    ## 6               0.0001239688            0.015101573                0.008367638
    ## 7               0.0003204930            0.026790723                0.015261943
    ## 8               0.0002817901            0.019266288                0.008491583
    ## 9               0.0001503719            0.022619893                0.012930504
    ## 10              0.0002991583            0.016273318                0.007032272
    ## 11              0.0002976799            0.016378640                0.008556172
    ## 12              0.0003396954            0.028776678                0.017984430
    ## 13              0.0003626576            0.028934894                0.015490686
    ## 14              0.0003442976            0.019189974                0.008692210
    ## 15              0.0003493378            0.023055199                0.014015012
    ## 16              0.0002774876            0.026625346                0.015543405
    ## 17              0.0004343452            0.018241538                0.008257457
    ## 18              0.0002157287            0.029644332                0.016779161
    ## 19              0.0005368837            0.007273601                0.010961177
    ## 20              0.0001891864            0.006289856                0.012942343
    ## 21              0.0011323354            0.010797243                0.022788792
    ##    Operating.Cash.Flow.Per.Share Free.Cash.Flow.Per.Share
    ## 2                    0.036407061              0.036663090
    ## 3                    0.030216458              0.020852009
    ## 4                    0.027609281              0.020202461
    ## 5                    0.017989514              0.011420368
    ## 6                    0.013839007              0.010757637
    ## 7                    0.011017727              0.011901998
    ## 8                    0.015460711              0.010101195
    ## 9                    0.008986813              0.008009765
    ## 10                   0.012280057              0.009269680
    ## 11                   0.015271399              0.011832329
    ## 12                   0.014236089              0.008546578
    ## 13                   0.015719668              0.009807483
    ## 14                   0.018190107              0.017561226
    ## 15                   0.011106746              0.010761673
    ## 16                   0.012781260              0.009790376
    ## 17                   0.010134058              0.013778100
    ## 18                   0.031770811              0.017713201
    ## 19                   0.025179793              0.023008316
    ## 20                   0.029558462              0.032707308
    ## 21                   0.015054740              0.016215072
    ## 
    ## Residual Deviance: 29075.42 
    ## AIC: 30235.42

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
data.frame(confusion.matrix$overall)
```

    ##                confusion.matrix.overall
    ## Accuracy                   2.307692e-01
    ## Kappa                      1.538366e-01
    ## AccuracyLower              2.099777e-01
    ## AccuracyUpper              2.525876e-01
    ## AccuracyNull               1.215255e-01
    ## AccuracyPValue             7.077183e-33
    ## McnemarPValue                       NaN

``` r
# Print validation stats of the model
data.frame(confusion.matrix$byClass)
```

    ##           Sensitivity Specificity Pos.Pred.Value Neg.Pred.Value Precision
    ## Class: 1   0.00000000   0.9960759      0.0000000      0.9883193 0.0000000
    ## Class: 2   0.00000000   0.9986945      0.0000000      0.9902913 0.0000000
    ## Class: 3   0.04761905   0.9940199      0.1818182      0.9739583 0.1818182
    ## Class: 4   0.00000000   0.9953177      0.0000000      0.9662338 0.0000000
    ## Class: 5   0.14102564   0.9680054      0.1896552      0.9550034 0.1896552
    ## Class: 6   0.31360947   0.8664731      0.2236287      0.9114504 0.2236287
    ## Class: 7   0.19310345   0.9486448      0.2800000      0.9191431 0.2800000
    ## Class: 8   0.15527950   0.9689755      0.3676471      0.9080460 0.3676471
    ## Class: 9   0.64893617   0.7034584      0.2323810      0.9354207 0.2323810
    ## Class: 10  0.06250000   0.9914469      0.4285714      0.9115334 0.4285714
    ## Class: 11  0.41818182   0.9102296      0.2628571      0.9533528 0.2628571
    ## Class: 12  0.04705882   0.9774282      0.1081081      0.9463576 0.1081081
    ## Class: 13  0.19780220   0.9546703      0.2142857      0.9501025 0.2142857
    ## Class: 14  0.29545455   0.9307745      0.2047244      0.9563380 0.2047244
    ## Class: 15  0.17647059   0.9652406      0.1475410      0.9717362 0.1475410
    ## Class: 16  0.00000000   0.9966689      0.0000000      0.9701686 0.0000000
    ## Class: 17  0.00000000   0.9993364      0.0000000      0.9741268 0.0000000
    ## Class: 18  0.26666667   0.9908616      0.2222222      0.9928058 0.2222222
    ## Class: 19  0.00000000   0.9987063      0.0000000      0.9993528 0.0000000
    ## Class: 20  0.00000000   0.9987038      0.0000000      0.9974110 0.0000000
    ## Class: 21  0.00000000   1.0000000            NaN      0.9974144        NA
    ##               Recall         F1   Prevalence Detection.Rate
    ## Class: 1  0.00000000        NaN 0.0116354234    0.000000000
    ## Class: 2  0.00000000        NaN 0.0096961862    0.000000000
    ## Class: 3  0.04761905 0.07547170 0.0271493213    0.001292825
    ## Class: 4  0.00000000        NaN 0.0336134454    0.000000000
    ## Class: 5  0.14102564 0.16176471 0.0504201681    0.007110537
    ## Class: 6  0.31360947 0.26108374 0.1092436975    0.034259858
    ## Class: 7  0.19310345 0.22857143 0.0937297996    0.018099548
    ## Class: 8  0.15527950 0.21834061 0.1040723982    0.016160310
    ## Class: 9  0.64893617 0.34221599 0.1215255333    0.078862314
    ## Class: 10 0.06250000 0.10909091 0.0930833872    0.005817712
    ## Class: 11 0.41818182 0.32280702 0.0711053652    0.029734971
    ## Class: 12 0.04705882 0.06557377 0.0549450549    0.002585650
    ## Class: 13 0.19780220 0.20571429 0.0588235294    0.011635423
    ## Class: 14 0.29545455 0.24186047 0.0568842922    0.016806723
    ## Class: 15 0.17647059 0.16071429 0.0329670330    0.005817712
    ## Class: 16 0.00000000        NaN 0.0297349709    0.000000000
    ## Class: 17 0.00000000        NaN 0.0258564964    0.000000000
    ## Class: 18 0.26666667 0.24242424 0.0096961862    0.002585650
    ## Class: 19 0.00000000        NaN 0.0006464124    0.000000000
    ## Class: 20 0.00000000        NaN 0.0025856496    0.000000000
    ## Class: 21 0.00000000         NA 0.0025856496    0.000000000
    ##           Detection.Prevalence Balanced.Accuracy
    ## Class: 1          0.0038784745         0.4980379
    ## Class: 2          0.0012928248         0.4993473
    ## Class: 3          0.0071105365         0.5208195
    ## Class: 4          0.0045248869         0.4976589
    ## Class: 5          0.0374919198         0.5545155
    ## Class: 6          0.1531997414         0.5900413
    ## Class: 7          0.0646412411         0.5708741
    ## Class: 8          0.0439560440         0.5621275
    ## Class: 9          0.3393665158         0.6761973
    ## Class: 10         0.0135746606         0.5269734
    ## Class: 11         0.1131221719         0.6642057
    ## Class: 12         0.0239172592         0.5122435
    ## Class: 13         0.0542986425         0.5762363
    ## Class: 14         0.0820943762         0.6131145
    ## Class: 15         0.0394311571         0.5708556
    ## Class: 16         0.0032320621         0.4983344
    ## Class: 17         0.0006464124         0.4996682
    ## Class: 18         0.0116354234         0.6287641
    ## Class: 19         0.0012928248         0.4993532
    ## Class: 20         0.0012928248         0.4993519
    ## Class: 21         0.0000000000         0.5000000

The model has a very low accuracy but is still better than the random
guess. For this case Positive Predictive Value is more important, since
false positives will be highly detrimental for the company and more
correct ratings (positive values) should be identified.

``` r
algorithm <- "Logistic.Regression"
save.class.acc.result(confusion.matrix$overall, algorithm)
save.class.pvv.result(confusion.matrix$byClass, algorithm)
```