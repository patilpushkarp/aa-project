# Data Preprocessing

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
library(stringr)
```

``` r
# Read the data
df <- read.csv("./../data/regression_data/input/player_data.csv")
```

``` r
# Get initial dimensions
dim(df)
```

    ## [1] 1122   45

``` r
names(df)
```

    ##  [1] "X"                "Name"             "Apps"             "Mins"            
    ##  [5] "Mins.Gm"          "Height"           "Weight"           "Age"             
    ##  [9] "Av.Rat"           "Gls"              "Gls.90"           "Shot.."          
    ## [13] "Pen.R"            "xG"               "Ch.C.90"          "Asts.90"         
    ## [17] "K.Ps.90"          "Pas.."            "Cr.C.A"           "Drb.90"          
    ## [21] "Distance"         "Hdr.."            "K.Tck"            "Fls"             
    ## [25] "Int.90"           "Clear"            "Con.90"           "xSv.."           
    ## [29] "Sv.."             "Svh"              "Svt"              "Svp"             
    ## [33] "Pens.Saved.Ratio" "PoM"              "Aer.A.90"         "Off"             
    ## [37] "Based"            "Tck.R"            "CA"               "Saves"           
    ## [41] "Saves.xSv."       "Gls.xG"           "Dist.Mins"        "Transfer.Value"  
    ## [45] "Value"

Following data preprocessing operations needs to be performed on the
data:

1.  Non-relevant columns will be removed which include *Based*, *Name*.
2.  Highly Correlated features will be removed.
3.  Some columns such as *transfer.value* needs special treatment.
4.  Remove columns with near zero variance.

``` r
## Remove unnecessary columns
remove.columns <- c("X", "Based", "Name")
df <- df[,!(names(df) %in% remove.columns)]

## Remove correlated columns
cor.columns <- c("Int.90", "Clear", "Svt", "Svp", "Svh", "Saves", "Pens.Saved.Ratio")
df <- df[, !(names(df) %in% cor.columns)]

## Split the transfer value
df[c("Lower.Transfer.Value", "Upper.Transfer.Value")] <- str_split_fixed(df$Transfer.Value ," - ", 2)

# Function to clean transfer value
clean.transfer.value <- function(x){
  if (grepl("€", x, fixed = TRUE)){
    x = sub("€","",as.character(x))
  }
  
  if (grepl("K", x, fixed = TRUE)){
    return (as.numeric(sub("K","",as.character(x))) * 1000)
  } else if (grepl("M", x, fixed = TRUE)){
    return (as.numeric(sub("M","",as.character(x))) * 1000000)
  } else{
    return (x)
  }
}

# Clean Upper limit of transfer value
df$Upper.Transfer.Value <- as.numeric(unlist(apply(df["Upper.Transfer.Value"], 1, clean.transfer.value)))
df$Upper.Transfer.Value[is.na(df$Upper.Transfer.Value)] <-0 
# Clean Lower limit of transfer value
df$Lower.Transfer.Value <- as.numeric(unlist(apply(df["Lower.Transfer.Value"], 1, clean.transfer.value)))
# Swap values if upper limit < lower limit
swap_index = rownames(df[df$Upper.Transfer.Value<df$Lower.Transfer.Value,]) # Find the indexes for which upper limit < lower limit
for (i in 1:length(swap_index)){
  # Iterate over indexes swapping values
  temp = df$Lower.Transfer.Value[i]
  df$Lower.Transfer.Value[i] = df$Upper.Transfer.Value[i]
  df$Upper.Transfer.Value[i] = temp
}
# Remove the original column
df <- df[, !(names(df) %in% c("Transfer.Value"))]

## Remove columns with near zero variance
nz.columns <- c("Pen.R", "Con.90", "xSv..", "Sv..", "Svh", "Svt", "Svp", "Pens.Saved.Ratio", "Saves", "Saves.xSv.")
df <- df[, !(names(df) %in% nz.columns)]
```

Write this preprocessed data to a file so that it can be used for
comparing models.

``` r
# Save the data with outliers
write.csv(df, "./../data/regression_data/intermediates/preprocessed_data.csv", row.names = FALSE)
```

## Outliers Analysis

Though individual features seems to have a lot of outliers, the reality
may not be the same. The data should be viewed holistically with the
effect of variables on each other.

``` r
## Check for outliers

# Create a model
glm.model <- glm(CA~., data=df)

# Get the outliers
outliers <- outlierTest(glm.model)
exclusion <- names(outliers[[1]])
exclusion <- as.numeric(unlist(exclusion))

df[exclusion,]
```

    ##     Apps Mins Mins.Gm Height Weight Age Av.Rat Gls Gls.90 Shot..   xG Ch.C.90
    ## 936   19 1710      90    196     84  24   7.33   2   0.11   0.33 0.98    0.58
    ##     Asts.90 K.Ps.90 Pas.. Cr.C.A Drb.90 Distance Hdr.. K.Tck Fls PoM Aer.A.90
    ## 936    0.05    0.26  0.94   0.66   0.26    202.7  0.85     3  16   3    12.68
    ##     Off Tck.R  CA   Gls.xG Dist.Mins Value Lower.Transfer.Value
    ## 936   2  0.87 143 2.040816  0.118538 2e+07                4e+07
    ##     Upper.Transfer.Value
    ## 936                    0

The outlier detection algorithm has detected outliers which can be
removed from the dataset.

``` r
# Remove the outliers
for(i in 1: length(exclusion))
{
  df = df[-exclusion[i],]
}
```

``` r
# Get the number of outliers
length(exclusion)
```

    ## [1] 1

``` r
# Save this new data to a file
# write.csv(df, "./../data/regression_data/intermediates/preprocessed_data_without_outliers.csv", row.names = FALSE)
```
