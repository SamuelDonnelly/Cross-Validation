ds.original <- read.csv(file = "test-data-set.txt", header = TRUE, sep = " ")
print(ds.original)

for (iternum in (1:3)){
    print(paste("Shuffle number:", iternum))
    ds.shuffled <- ds.original[sample(nrow(ds.original)), ]
    print(ds.shuffled)
}

library(MASS)
dataset.obj <- as.data.frame(Boston)
summary(dataset.obj)
## change the variable chas to factor, since it's a dummy variable
dataset.obj$chas <- as.factor(dataset.obj$chas)
## verify
summary(dataset.obj$chas)

## The overall k-fold-based CV approach
samplesize <- nrow(dataset.obj)
numfolds <- 10 # we're setting k = 10

quotient <- samplesize %/% numfolds # the %/% operator returns the quotient of a division
remainder <- samplesize %% numfolds # the %% operator returns the remainder of a division

vct.sizes <- rep(quotient, numfolds) # create a vector representing the initial subsets with size = quotient
if(remainder > 0){
    for(i in 1:remainder){
        vct.sizes[i] <- vct.sizes[i] + 1 # for the "remainder" number of subsets, add one to their size
    }
}

print(paste("K:", 10, "n:", samplesize))
print(vct.sizes)

startval <- 1
endval <- nrow(dataset.obj)/2

model <- glm(crim ~ indus + dis + medv,
             data = dataset.obj[-(startval:endval), ])
pred.vals <- predict(model, newdata=dataset.obj[startval:endval, ],
                     type="response")
rmse <- sqrt(mean((dataset.obj$crim[startval:endval] - pred.vals)^2))

set.seed(112)
dataset.obj <- dataset.obj[sample(nrow(dataset.obj)), ]

## create the vector to hold RMSE values
vct.rmses <- numeric(numfolds)

startval <- 1
for(kth in (1:numfolds)){
    endval <- vct.sizes[kth] + startval - 1

    model <- glm(crim ~ indus + dis + medv,
                 data = dataset.obj[-(startval:endval), ])
    pred.vals <- predict(model,
                         newdata = dataset.obj[startval:endval, ],
                         type="response")
    ## compute the RMSE
    rmse <- sqrt(mean((dataset.obj$crim[startval:endval] - pred.vals)^2))
    ## store the current fold's RMSE
    vct.rmses[kth] <- rmse
    ## modify the start value to correspond to the next fold
    startval <- endval + 1
}

## Compute the overall RMSE
overall.rmse <- mean(vct.rmses)
print(paste("For the model crim ~ indus + dis + medv, the overall 10-fold CV RMSE is:",
            round(overall.rmse, 6)))

set.seed(112)
dataset.obj <- dataset.obj[sample(nrow(dataset.obj)), ]

## create the vector to hold RMSE values
vct.rmses <- numeric(numfolds)

startval <- 1
for(kth in (1:numfolds)){
    endval <- vct.sizes[kth] + startval - 1

    model <- glm(crim ~ indus + dis + medv + age,
                 data = dataset.obj[-(startval:endval), ])
    pred.vals <- predict(model,
                         newdata = dataset.obj[startval:endval, ],
                         type="response")
    ## compute the RMSE
    rmse <- sqrt(mean((dataset.obj$crim[startval:endval] - pred.vals)^2))
    ## store the current fold's RMSE
    vct.rmses[kth] <- rmse
    ## modify the start value to correspond to the next fold
    startval <- endval + 1
}

## Compute the overall RMSE
overall.rmse <- mean(vct.rmses)
print(paste("For the model crim ~ indus + dis + medv + age, the overall 10-fold CV RMSE is:",
            round(overall.rmse, 6)))

getCoeffs <- function(dataset, indices, formula){
    d <- dataset[indices, ]
    fit <- glm(formula, data=d)
    return(coef(fit))
}
library(boot)
set.seed(122)
results <- boot(data=Boston,
                statistic=getCoeffs,
                R=1000,
                formula = crim ~ indus + dis + medv)


print("Coefficients of the model")
print(coef(model))
print("Estimated coefficient values from bootstrap")
print(results)
prednames <- c("indus", "dis", "medv")
# Since we have four coefficients, we will iterate over the computations of CIs, one at a time
for(i in 2:4){
    print(paste("The confidence interval for", prednames[i-1], "is:"))
    print(boot.ci(results, conf=0.95, index=i, type="basic"))
}
