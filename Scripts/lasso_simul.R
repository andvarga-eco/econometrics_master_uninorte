library(glmnet)
library(MASS)
library(psych)
library(hdm)


# Generamos datos
## n=40, p=3

mu<-c(0,0,0)
sigma<-matrix(c(1,0.5,0.5,0.5,1,0.5,0.5,0.5,1),3,3)
print(sigma)
n<-40
set.seed(12345)
X<-mvrnorm(n,mu,sigma)
head(X)

e<-rnorm(40,0,3)
y<-2+1*X[,1]+e

data<-cbind(X,y)
describe(data)
cor(data)

## Lasso, n=40,p=3,k=5
###  k=5, 5-fold cross validation para lambda
set.seed(10101)
fit.lasso<-cv.glmnet(X,y, nfolds = 5)
print(fit.lasso)
plot(fit.lasso)
fit.lasso$lambda.min
coef(fit.lasso,s="lambda.min")

### in sample fit

yhat<-predict(fit.lasso,newx=X,s="lambda.min")
lasso_res<-y-yhat
mse_l<-mean(lasso_res^2)

### Split sample

set.seed(10101) # to make the results replicable (we will generate random numbers)
random <- sample(1:length(y), floor(n * 4 / 5)) # draw (4/5)*n random numbers from 1 to n without replacing
train <- data[random, ]
test <- data[-random, ]

X_train<-train[,1:3]
X_test<-test[,1:3]
y_train<-train[,4]
y_test<-test[,4]

set.seed(10101)
fit.lasso<-cv.glmnet(X_train,y_train, nfolds = 5)
print(fit.lasso)
plot(fit.lasso)
fit.lasso$lambda.min
coef(fit.lasso,s="lambda.min")

yhat_s<-predict(fit.lasso,newx=X_train,s="lambda.min")
lasso_res_s<-y_train-yhat_s
mse_l_s<-mean(lasso_res_s^2)

yhat_p<-predict(fit.lasso,newx=X_test,s="lambda.min")
lasso_res_p<-y_test-yhat_p
mse_l_p<-mean(lasso_res_p^2)


### Post-Lasso

fit.postlasso<-rlasso(X,y,post=TRUE)


