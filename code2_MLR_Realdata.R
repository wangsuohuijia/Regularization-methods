##  statistical learning ## assignment 2 

library(glmnet)
library(e1071)
library(plyr)

#### data
data <- data.frame(read.csv("energydata_complete.csv"))

#converting date to numeric
data[,1] <-as.numeric(as.POSIXct(data[,1], format="%Y-%m-%d  %H:%M:%S"))

#removing rows that are incomplete or have null values
data <- data[complete.cases(data)== TRUE,]

set.seed(57)
train_index <- sample (1: nrow(data), nrow(data)*0.7)
test_index <- (-train_index )
data.train <-data[train_index,]
data.test <-data[test_index,]

# define data
x.train <- model.matrix(Appliances~., data.train)[, -1]
y.train <- data.train$Appliances
x.test <- model.matrix(Appliances~., data.test)[, -1]
y.test <- data.test$Appliances
grid <- 10^ seq (10,-2, length =100)


#### multiple linear regression
lm.fit <- lm(Appliances~., data.train)
summary(lm.fit)
prediction.lm <- predict(lm.fit, data.test)
# plot comparison between prediction and ture value
plot(data.test$Appliances, ylab='Appliances, energy use in Wh', type='l', main='Comparison of Prediction & True Value with MLR', col='black')
lines(prediction.lm, col = 'red')
legend('topright', legend=c('True Value', 'Prediction'), fill =c('black', 'red'))
# calculate and plot residuals
residuals.lm <- prediction.lm - data.test$Appliances
plot(residuals.lm, ylab='Residual', main = "Residuals with MLR")
# calculate mse
mse.lm <- mean((prediction.lm - data.test$Appliances)^2)
mse.lm


#### ridge regression
# ridge regression without cross validataion
ridge.mod <- glmnet(x.train, y.train, alpha=0, lambda=grid, thresh =1e-12)
prediction.ridge <- predict(ridge.mod, s=4, newx=x.test)
# plot comparison between prediction and ture value
plot(data.test$Appliances, type='l', main='Comparison of Prediction & True Value with Ridge', col='black')
lines(prediction.ridge, col = 'red')
legend('topright', c('True Value', 'Prediction'), fill =c('black', 'red'))
# plot residuals
residuals.ridge <- prediction.ridge - y.test
plot(residuals.ridge, ylab='Residual', main = "Residuals with Ridge")
# calculate mse
mse.ridge <- mean((prediction.ridge - y.test)^2)
mse.ridge

# ridge regression with cross validation
set.seed(1)
cv.ridge.mod <- cv.glmnet(x.train, y.train, alpha = 0)
bestlam.ridge <- cv.ridge.mod$lambda.min
prediction.cv.ridge <- predict(cv.ridge.mod, s=bestlam.ridge, newx=x.test)
# plot comparison between prediction and ture value
plot(data.test$Appliances, type='l', main='Comparison of Prediction & True Value with Ridge.CV', col='black')
lines(prediction.cv.ridge, col = 'red')
legend('topright', c('True Value', 'Prediction'), fill =c('black', 'red'))
# plot residuals
residuals.cv.ridge <- prediction.cv.ridge - y.test
plot(residuals.cv.ridge, ylab='Residual', main = "Residuals with Ridge.CV")
mse.cv.ridge <- mean((prediction.cv.ridge - y.test)^2)

# comparison between ridge regreesion without and with cross validation
mse.ridge
mse.cv.ridge

# use the best model (with cross validataion) to get coefficients
ridge.coef <- predict(cv.ridge.mod, type = 'coefficients', s = bestlam.ridge)[1:29,]
ridge.coef


#### lasso regression
# lasso regression without cross validataion
lasso.mod <- glmnet(x.train, y.train, alpha = 1, lambda = grid)
prediction.lasso <- predict(lasso.mod, s=4, newx=x.test)
# plot comparison between prediction and ture value
plot(data.test$Appliances, type='l', ylab='Appliances, energy use in Wh', main='Comparison of Prediction & True Value with LASSO', col='black')
lines(prediction.lasso, col = 'red')
legend('topright', c('True Value', 'Prediction'), fill =c('black', 'red'))
# plot residuals
residuals.lasso <- prediction.lasso - y.test
plot(residuals.lasso, ylab='Residual', main = "Residuals with LASSO")
# calculate mse
mse.lasso <- mean((prediction.lasso - y.test)^2)

# lasso regression with cross validataion
set.seed(1)
cv.lasso.mod <- cv.glmnet(x.train, y.train, alpha = 1)
bestlam.lasso <- cv.lasso.mod$lambda.min
prediction.cv.lasso <- predict(cv.lasso.mod, s=bestlam.lasso, newx = x.test) 
# plot comparison between prediction and ture value
plot(data.test$Appliances, type='l', ylab='Appliances, energy use in Wh', main='Comparison of Prediction & True Value with LASSO.CV', col='black')
lines(prediction.cv.lasso, col = 'red')
legend('topright', c('True Value', 'Prediction'), fill =c('black', 'red'))
# plot residuals
residuals.cv.lasso <- prediction.cv.lasso - y.test
plot(residuals.cv.lasso, ylab='Residual', main = "Residuals with LASSO.CV")
# calculate mse
mse.cv.lasso <- mean((prediction.cv.lasso-y.test)^2)

# comparison between lasso regreesion without and with cross validation
mse.lasso
mse.cv.lasso

# use the best model (with cross validataion) to get coefficients
lasso.coef <- predict(cv.lasso.mod, type = 'coefficients', s = bestlam.lasso)[1:29,]
lasso.coef


#### elastic net
#finding the best lambda for the elastic net between l0 and l1 penalties
elastic.msqVector <- rep(100,101)
cv.elastic.msqVector <- rep(100,101)

for (i in 0:100) {
  elastic.lambda <-i/100
  
  # elastic net regression without cross validataion
  set.seed(1)
  elastic.mod <- glmnet(x.train, y.train, alpha = elastic.lambda, lambda=grid)
  prediction.elastic <- predict(elastic.mod, s=elastic.lambda, newx = x.test) 
  mse.elastic <- mean((prediction.elastic-y.test)^2)
  elastic.msqVector[i+1] <- mse.elastic
  
  # elastic net regression with cross validataion
  set.seed(1)
  cv.elastic.mod <- cv.glmnet(x.train, y.train, alpha = elastic.lambda)
  bestlam <- cv.elastic.mod$lambda.min
  prediction.cv.elastic <- predict(cv.elastic.mod, s=bestlam, newx = x.test) 
  mse.cv.elastic <- mean((prediction.cv.elastic-y.test)^2)
  cv.elastic.msqVector[i+1] <- mse.cv.elastic
}

# plot vector of the MSE of all the lambdas which were trialled
plot(elastic.msqVector)
plot(cv.elastic.msqVector)

# elastic net regression without cross validataion
# determine best lambda
elastic.minlambdaindex <- which(elastic.msqVector == min(elastic.msqVector))[1]-1
elastic.lambda <-elastic.minlambdaindex/100
# build model and make predictions
elastic.mod <- glmnet(x.train,y.train, alpha=elastic.lambda, lambda=4 ,thresh =1e-12)
prediction.elastic <- predict(elastic.mod, newx=x.test)
# plot comparison between prediction and ture value
plot(data.test$Appliances, type='l', ylab='Appliances,energy use in Wh', main='Comparison of Prediction & True Value with ElasticNet', col='black')
lines(prediction.elastic, col = 'red')
legend('topright', c('True Value', 'Prediction'), fill =c('black', 'red'))
# plot residuals
residuals.elastic <- prediction.elastic - y.test
plot(residuals.elastic, ylab='Residual', main = "Residuals with ElasticNet")
# calculate mse
mse.elastic <- mean((prediction.elastic - y.test)^2)

# elastic net regression with cross validataion
# determine best lambda
cv.elastic.minlambdaindex <- which(cv.elastic.msqVector == min(cv.elastic.msqVector))[1]-1
cv.elastic.lambda <-cv.elastic.minlambdaindex/100
bestlam.cv.elastic <- cv.elastic.mod$lambda.min
#build model and make predictions
set.seed(1)
cv.elastic.mod <- cv.glmnet(x.train, y.train, alpha = cv.elastic.lambda)
prediction.cv.elastic <- predict(cv.elastic.mod, s=bestlam.cv.elastic, newx = x.test) 
# plot comparison between prediction and ture value
plot(data.test$Appliances, type='l', ylab='Appliances,energy use in Wh', main='Comparison of Prediction & True Value with ElasticNet.CV', col='black')
lines(prediction.cv.elastic, col = 'red')
legend('topright', c('True Value', 'Prediction'), fill =c('black', 'red'))
# plot residuals
residuals.cv.elastic <- prediction.cv.elastic - y.test
plot(residuals.cv.elastic, ylab='Residual', main = "Residuals with ElasticNet.CV")
# calculate mse
mse.cv.elastic <- mean((prediction.cv.elastic-y.test)^2)

# comparison between elastic net regreesion without and with cross validation
mse.elastic
mse.cv.elastic

# use the best model (with cross validataion) to get coefficients
elastic.coef <- predict(cv.elastic.mod, type = 'coefficients', s = bestlam.cv.elastic)[1:29,]
elastic.coef


#### comparison of all algorithms
mse.lm
mse.ridge
mse.cv.ridge
mse.lasso
mse.cv.lasso
mse.elastic
mse.cv.elastic

