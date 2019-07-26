#+eval=FALSE

## Load libraries
if(!require("randomForest")) install.packages("randomForest", repos = "http://cran.us.r-project.org")
if(!require("corrgram")) install.packages("corrgram", repos = "http://cran.us.r-project.org")
if(!require("caret")) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require("reshape2")) install.packages("reshape2", repos = "http://cran.us.r-project.org")
if(!require("GGally")) install.packages("reshape2", repos = "http://cran.us.r-project.org")

library(corrgram)
library(randomForest)
library(GGally)
library(caret)
library(tidyverse)
library(ggplot2)
library(reshape2)

set.seed(1234)

## path to the  file. 
file = "C:\\Users\\alanc\\Desktop\\UW Data Analysis Techniques for Descision Making\\Data Mining and Predictive Analytics\\kc_house_data.csv"

## read in file
house = read_csv(file)

summary(house)

# convert columns to factors
house = house %>% mutate(waterfront_fac = as.factor(waterfront), 
                        view_fac = as.factor(view), 
                        zipcode_fac = as.factor(zipcode))

# remove variables not needed for analysis
house$id <- NULL
house$zipcode <- NULL

# view dataset
summary(house)
glimpse(house)

# exploratory data analysis
ggpairs(house[,c(3:7,2)])

ggpairs(house[,c(10:12,16:17,2)])

#top correlations
cor_level <- .7
correlationMatrix <- cor(house[,2:19])
cor_melt <- arrange(melt(correlationMatrix),desc(value))

# filter duplicate rows and for correlation values greater than .7
dplyr::filter(cor_melt,
              row_number() %% 2 == 0,
              Var2 != "sqft_living15",
              Var2 != "sqft_lot15",
              value > cor_level,
              value != 1)

#show variables that correlate to price only
cor_level <- .5
dplyr::filter(cor_melt,
              row_number() %% 2 == 0,
              Var1 != "sqft_living15" & Var2 != "sqft_living15",
              Var1 != "sqft_lot15" & Var2 != "sqft_lot15",
              Var1 == "price" | Var2 == "price",
              value > cor_level,
              value != 1)

#remove variables that logically don't contribute to a good model
house$sqft_living15 <- NULL
house$sqft_lot15 <- NULL
house$waterfront <- NULL
house$view <- NULL
house$zipcode_fac <- NULL

# view dataset
glimpse(house)

# compute random forest model and view model
tree_model <- randomForest(price ~ ., data=house, importance = TRUE)
summary(tree_model)

plot(tree_model)

# variable importance of random forest model
var_imp <- varImp(tree_model)
var_imp
varImpPlot(tree_model)

print(tree_model)

# compute linear model and view confidence intervals
lm_model <- lm(price ~ ., data = house)
summary(lm_model)
confint(lm_model)

# cross validation of linear model and random forest model
train_control<- trainControl(method="cv", number=3, savePredictions = TRUE)
model <- train(price ~ ., data=house, trControl=train_control, method="lm")
model

train_control<- trainControl(method="cv", number=3, savePredictions = TRUE)
model <- train(price ~ ., data=house, trControl=train_control, method="rf")
model

# new random forest model
# create features
house = house %>% mutate(sqft_living.sqrt = sqrt(sqft_living),
                          sqft_living.log = log(sqft_living),
                          price.log = log(price))

# numeric columns to be scaled
cols = c('bedrooms',
         'bathrooms',
         'sqft_living',
         'sqft_lot',
         'floors',
         'condition',
         'grade',
         'sqft_above',
         'sqft_living.sqrt',
         'sqft_living.log')
cols

# scale numeric columns
house[,cols] = sapply(house[,cols], scale)

# calculate outlier price with interquartile range
outlierprice <- quantile(house$price, .75) + ((quantile(house$price, .75) - quantile(house$price, .25)) * 1.5)

# calculate new random forest model to predict log of price
new_rf_model <- randomForest(price.log ~ bedrooms + bathrooms + sqft_living +
                                     sqft_lot + floors + waterfront_fac + view_fac + condition +
                                     grade + sqft_above + sqft_living.sqrt + sqft_living.log,
                             data=filter(house,price<outlierprice))
new_rf_model

summary(new_rf_model)

# create prediction data frame
to_predict <- house[0,]
to_predict[1,]$sqft_living <- 4000
to_predict[1,]$sqft_above <- 4000
to_predict[1,]$sqft_lot <- 5000
to_predict[1,]$bedrooms <- 4
to_predict[1,]$bathrooms <- 3
to_predict[1,]$floors <- 2
to_predict[1,]$waterfront_fac <- 0
to_predict[1,]$view_fac <- 1
to_predict[1,]$condition <- 5
to_predict[1,]$grade <- 7
to_predict[1,]$sqft_living.sqrt <- sqrt(4000)
to_predict[1,]$sqft_living.log <- log(4000)

# random forest prediction price
exp(predict(new_rf_model, newdata=to_predict))

# calculate new linear model for predicting log of price
new_lm_model = lm(price.log ~ bedrooms + bathrooms + sqft_living +
                          sqft_lot + floors + waterfront_fac + view_fac + condition +
                          grade + sqft_above + sqft_living.sqrt + sqft_living.log,
                  data=filter(house,price<outlierprice))

# linear model prediction price
# (training data is scaled while prediction
#  data is not resulting in an inaccurate prediction)
predict(new_lm_model, newdata=to_predict)










