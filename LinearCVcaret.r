library(caret) #Classification And REgression Training
library(glmnet) #Fits ridege and lasso
library(MASS)
library(tidyverse)
library(kableExtra)

## REad data
data <- as.data.frame(Boston)
data$chas <- as.factor(data$chas)

## ID non/near zeros
nzv <- nearZeroVar(data, saveMetrics = TRUE)
nzv %>% kable(digits = 3) %>% kable_classic(bootstrap_options = "condensed", 
                                            html_font = "Cambria" )

## indicate which var. is outcome
# indicate proportion to use in training-testing
set.seed(777)
trainIndices <- createDataPartition(data$medv, 
                                    p = 0.7, 
                                    list = FALSE, 
                                    times = 1)
training <- data[trainIndices,]
holdout <- data[-trainIndices,]

preProcValues <- preProcess(training, method = c("center", "scale"))
## Next, create the scaled+centered of the training+testing subset of the 
## dataset
trainTransformed <- predict(preProcValues, training) 
## apply the same scaling and centering on the holdout set, too
holdoutTransformed <- predict(preProcValues, holdout)

## ID optimal model
fitControl <- trainControl(
  method = "repeatedcv", ## perform repeated k-fold CV
  number = 10, ## 10-fold CV
  ## repeated ten times
  repeats = 10)

## fit ridge
grid <- expand.grid(lambda = 10 ^ seq(10, -2, length = 100),
                    alpha = 0) # grid of values lambda = 10^10 to 10^-2 covering
# full range of scenarios from the null model.
# alpha=1=lasso, 2=ridge 
ridgefit <- train(medv ~ ., data = trainTransformed, 
                  method = "glmnet",
                  trControl = fitControl, 
                  verbose = FALSE, 
                  tuneGrid = grid) # hyperparam

### Ridge Scoring & variable importance

predvals.r <- predict(ridgefit, holdoutTransformed)
ridge.perf <- postResample(pred = predvals.r, obs = holdoutTransformed$medv)[-2]
ridge.import <- varImp(ridgefit)


## (1b) Fit Lasso 

## grid for lasso with alpha = 1
grid2 <- expand.grid(lambda = 10 ^ seq(10, -2, length = 100),
                     alpha = 1)

lassofit <- train(medv ~ ., data = trainTransformed, 
                  method = "glmnet",
                  trControl = fitControl, 
                  verbose = FALSE, 
                  tuneGrid = grid2)


### Lasso Scoring & variable importance

predvals.l <- predict(lassofit, holdoutTransformed)
lasso.perf <- postResample(pred = predvals.l, obs = holdoutTransformed$medv)[-2]
lasso.import <- varImp(lassofit) 


## (1c) Multiple Linear

## Fit lm model
mlfit <- train(medv ~ ., data = trainTransformed, 
               method = "lm",
               trControl = fitControl)


### Multiple Linear Scoring & variable importance

predvals.ml <- predict(mlfit, holdoutTransformed)
ml.perf <- postResample(pred = predvals.ml, obs = holdoutTransformed$medv)[-2]
ml.import <- varImp(mlfit)


## (2) Generate Performance DF

df <- tibble(Model = c("Ridge", "Lasso", "MLR"), RSME = c(ridge.perf[1], 
                                                          lasso.perf[1],
                                                          ml.perf[1]), MAE = 
               c(ridge.perf[2], lasso.perf[2],ml.perf[2]), Average= (RSME+MAE)/2
)
df %>%
  kable(caption = "70/30 Performance Table", digits=3) %>%
  kable_classic(bootstrap_options = "condensed", html_font = "Cambria", 
                position = "left", full_width = F)



  ## (3) Variable Importance
  
## Make df of importance variables
## rownames being weird, add them in manually
Predictors <- c("crim", "zn", "indus", "chas", "nox", "rm", "age", "dis", 
                "rad", "tax", "ptratio", "black", "lstat")
## df with Predictors, ridge, lasso, and mlr
df <- tibble(Predictors = Predictors, Ridge = c(ridge.import$importance[ ,1]), 
             Lasso = c(lasso.import$importance[ ,1]), MLR = 
               (ml.import$importance[ ,1]))
## add average column
df$Average <- (df$Ridge + df$Lasso + df$MLR)/3
## print table
df[with(df, order (-Average)), ] %>% kable(caption = 
                                             "Variable Importance 70/30", 
                                           digits=3) %>%
  kable_classic(bootstrap_options = "condensed", html_font = "Cambria", 
                position = "left", full_width = F)



### (5) Set data split 90/10

set.seed(777)
trainIndices <- createDataPartition(data$medv, 
                                  p = 0.9, 
                                  list = FALSE, 
                                  times = 1)
training <- data[trainIndices,]
holdout <- data[-trainIndices,]

preProcValues <- preProcess(training, method = c("center", "scale"))
## Next, create the scaled+centered of the training+testing subset of the data
trainTransformed <- predict(preProcValues, training) 
## apply the same scaling and centering on the holdout set, too
holdoutTransformed <- predict(preProcValues, holdout)

## Identify optimal model
fitControl <- trainControl(
  method = "repeatedcv", ## perform repeated k-fold CV
  number = 10, ## 10-fold CV
  ## repeated ten times
  repeats = 10)


### 5(a) Fit ridge, lasso, and mlr 

## set grid for ridge
grid <- expand.grid(lambda = 10 ^ seq(10, -2, length = 100),
                    alpha = 0) # grid of values lambda = 10^10 to 10^-2 covering
                               # full range of scenarios from the null model.
                               # alpha=1=lasso, 2=ridge 
## fit ridge model
ridgefit <- train(medv ~ ., data = trainTransformed, 
                 method = "glmnet",
                 trControl = fitControl, 
                 verbose = FALSE, 
                 tuneGrid = grid) # hyperparam

## set grid for lasso
grid2 <- expand.grid(lambda = 10 ^ seq(10, -2, length = 100),
                    alpha = 1)

## fit lasso model
lassofit <- train(medv ~ ., data = trainTransformed, 
                 method = "glmnet",
                 trControl = fitControl, 
                 verbose = FALSE, 
                 tuneGrid = grid2)

## fit mlr model
mlfit <- train(medv ~ ., data = trainTransformed, 
                 method = "lm",
                 trControl = fitControl)


### 5(b) Generate Performance Table

## ridge performance and importance
predvals.r <- predict(ridgefit, holdoutTransformed)
ridge.perf <- postResample(pred = predvals.r, obs = holdoutTransformed$medv)[-2]
ridge.import <- varImp(ridgefit)

## lasso performance and importance
predvals.l <- predict(lassofit, holdoutTransformed)
lasso.perf <- postResample(pred = predvals.l, obs = holdoutTransformed$medv)[-2]
lasso.import <- varImp(lassofit) 

## mlr performance and importance
predvals.ml <- predict(mlfit, holdoutTransformed)
ml.perf <- postResample(pred = predvals.ml, obs = holdoutTransformed$medv)[-2]
ml.import <- varImp(mlfit)

## generate performance table
df <- tibble(Model = c("Ridge", "Lasso", "MLR"), RSME = 
               c(ridge.perf[1], lasso.perf[1],ml.perf[1]), MAE = 
               c(ridge.perf[2], lasso.perf[2],ml.perf[2]), Average =(RSME+MAE)/
               2)
df %>%
  kable(caption = "90/10 Performance Table", digits=3) %>%
  kable_classic(bootstrap_options = "condensed", html_font = "Cambria", 
                position = "left", full_width = F)


### 5(c) Variable Importance

## Make df of importance variables
## rownames being weird, add them in manually
Predictors <- c("crim", "zn", "indus", "chas", "nox", "rm", "age", "dis", 
                "rad", "tax", "ptratio", "black", "lstat")
## df with Predictors, ridge, lasso, and mlr
df <- tibble(Predictors = Predictors, Ridge = c(ridge.import$importance[ ,1]), 
             Lasso = c(lasso.import$importance[ ,1]), MLR = 
               (ml.import$importance[ ,1]))
## add average column
df$Average <- (df$Ridge + df$Lasso + df$MLR)/3
## print table
df[with(df, order (-Average)), ] %>% 
  kable(caption = "Variable Importance 90/10", digits=3) %>%
   kable_classic(bootstrap_options = "condensed", html_font = "Cambria",
                 position = "left", full_width = F)