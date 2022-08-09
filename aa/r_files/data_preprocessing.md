# Data Preprocessing

Before modelling data, let us utilize insights from our EDA notebook to
clean up the data.

``` r
# Load libraries
library(caret)
```

    ## Loading required package: ggplot2

    ## Loading required package: lattice

``` r
library(mltools)
library(data.table)
library(car)
```

    ## Loading required package: carData

``` r
library(dplyr)
```

    ## 
    ## Attaching package: 'dplyr'

    ## The following object is masked from 'package:car':
    ## 
    ##     recode

    ## The following objects are masked from 'package:data.table':
    ## 
    ##     between, first, last

    ## The following objects are masked from 'package:stats':
    ## 
    ##     filter, lag

    ## The following objects are masked from 'package:base':
    ## 
    ##     intersect, setdiff, setequal, union

``` r
# Read the data
df <- read.csv("./../data/classification_data/input/corporate_credit_rating.csv")
```

Following data preprocessing operations needs to be performed on the
data:

1.  Correct the data types of the columns. Two columns needs correction
    which are *CIK* and *SIC Code*.
2.  There were 3 credit rating agencies for which there is insufficient
    data. The data present with respect to these 3 agencies will be
    removed.
3.  Apart from *Net Margin*, all the other correlated columns will be
    removed.
4.  All irrelevant columns will be removed such as *SIC Code*, *CIK*,
    *Corporation Name*, *Ticker*, and *Rating Date.* These columns will
    not provide any information to the model even if they are encoded to
    numeric values.
5.  The target class for which sufficient data is not present will be
    removed.
6.  All the categorical variables remaining will be either one-hot
    encoded or label encoded. Following columns will be encoded:
    1.  One-hot encoding:

        1.  Sectors

        2.  Rating Agencies

    2.  Label encoding:

        1.  Rating

``` r
## Preprocessing

# Correct the data types of columns
df$CIK <- as.factor(df$CIK)
df$SIC.Code <- as.factor(df$SIC.Code)

# Remove data of under-represented rating agencies
rating.agency.freq <- data.frame(table(df$Rating.Agency))
nreq.rating.agencies <- rating.agency.freq[order(rating.agency.freq$Freq, decreasing = TRUE),]
nreq.rating.agencies <- tail(nreq.rating.agencies, 3)
nreq.rating.agencies <- nreq.rating.agencies$Var1
df <- df[!df$Rating.Agency %in% nreq.rating.agencies,]

# Remove correlated columns
num.data <- df %>% dplyr::select(where(is.numeric))
cor_mat <- cor(num.data)
index <- findCorrelation(cor_mat, .75)
col.remove <- colnames(cor_mat)[index]
col.remove <- col.remove[-length(col.remove)]
df <- df[!names(df) %in% col.remove]

# Remove ID columns and other irrelevant columns
df <- df[!names(df) %in% c("SIC.Code", "CIK", "Ticker", "Corporation", "Rating.Date")]

# Remove under-represented target variable observations
ratings.count.df <- data.frame(table(df$Rating))
rtl.data <- ratings.count.df[ratings.count.df$Freq<10,]
df <- df[!df$Rating %in% rtl.data$Var1,]

# Encode categorical columns
# Encode Rating Agencies and Sectors
df$Rating.Agency <- as.factor(df$Rating.Agency)
df$Sector <- as.factor(df$Sector)
new.df <- one_hot(as.data.table(df), cols=c("Rating.Agency", "Sector"))

# Ecode Target Column
rating.order <- c("AAA", "AA+", "AA", "AA-", "A+", "A", "A-", "BBB+", "BBB", "BBB-", "BB+", "BB", "BB-", "B+", "B", "B-", "CCC+", "CCC", "CCC-", "CC", "C", "D")
new.df$Rating <- as.factor(new.df$Rating)
rating.factors <- factor(new.df$Rating, levels=rating.order)
rating.factors <- as.numeric(rating.factors)
new.df$Rating <- rating.factors
```

Write this preprocessed data to a file so that it can be used for
comparing models.

``` r
# Save the data with outliers
write.csv(new.df, "./../data/classification_data/intermediates/preprocessed_data.csv", row.names = TRUE)
```

## Outliers Analysis

Though individual features seems to have a lot of outliers, the reality
may not be the same. The data should be viewed holistically with the
effect of variables on each other.

``` r
## Check for outliers

# Create a model
glm.model <- glm(Rating~., data=new.df)

# Get the outliers
outliers <- outlierTest(glm.model)
exclusion <- names(outliers[[1]])
exclusion <- as.numeric(unlist(exclusion))

new.df[exclusion,]
```

    ##    Rating.Agency_Egan-Jones Ratings Company Rating.Agency_Fitch Ratings
    ## 1:                                        0                           0
    ##    Rating.Agency_Moody's Investors Service
    ## 1:                                       0
    ##    Rating.Agency_Standard & Poor's Ratings Services Rating Binary.Rating
    ## 1:                                                1     17             0
    ##    Sector_BusEq Sector_Chems Sector_Durbl Sector_Enrgy Sector_Hlth Sector_Manuf
    ## 1:            0            0            1            0           0            0
    ##    Sector_Money Sector_NoDur Sector_Other Sector_Shops Sector_Telcm
    ## 1:            0            0            0            0            0
    ##    Sector_Utils Current.Ratio Long.term.Debt...Capital Debt.Equity.Ratio
    ## 1:            0        1.2381                    171.5           -1.0059
    ##    Gross.Margin Net.Profit.Margin Asset.Turnover ROE...Return.On.Equity
    ## 1:      12.0397            0.3399         1.2261                 -2.737
    ##    Return.On.Tangible.Equity ROA...Return.On.Assets ROI...Return.On.Investment
    ## 1:                   -1.9244                 0.9726                   466.6667
    ##    Operating.Cash.Flow.Per.Share Free.Cash.Flow.Per.Share
    ## 1:                        6.4777                   6.9808

The outlier detection algorithm has detected 1 outlier which can be
removed from the dataset.

``` r
# Remove the outliers
for(i in 1: length(exclusion))
{
  new.df = new.df[-exclusion[i],]
}
```

``` r
# Save this new data to a file
write.csv(new.df, "./../data/classification_data/intermediates/preprocessed_data_without_outliers.csv", row.names = TRUE)
```
