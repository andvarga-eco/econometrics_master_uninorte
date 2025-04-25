library(dplyr)
library(caret)
library(rpart)
library(RColorBrewer)
library(rpart.plot)
library(randomForest)

file <- "https://raw.githubusercontent.com/CausalAIBook/MetricsMLNotebooks/main/data/wage2015_subsample_inference.csv"
data <- read.csv(file)

#Objetivo: modelo predictivo para lwage

## 1. Split sample

data$occ2<-factor(data$occ2) # definir como factor
data$ind2<-factor(data$ind2) # definir como factor

set.seed(1234)
training <- sample(nrow(data), nrow(data) * (3 / 4), replace = FALSE)
data_train <- data[training, ]
data_test <- data[-training, ]

## 2. Regression tree

basico<-lwage ~ (sex + exp1 + shs + hsg + scl + clg + mw + so + we + occ2 + ind2)

tree.fit.basic<-rpart(basico,data=data_train,minbucket=5,cp=0.01)
printcp(tree.fit.basic)
rsq.rpart((tree.fit.basic))

bestcp <- tree.fit.basic$cptable[which.min(tree.fit.basic$cptable[, "xerror"]), "CP"]
bestcp

fit_prunedtree <- prune(tree.fit.basic, cp = bestcp)
rsq.rpart(fit_prunedtree)
rpart.plot(fit_prunedtree)

y_test<-data_test$lwage
yhat_pt <- predict(fit_prunedtree, newdata = data_test)
mse_pt <- summary(lm((y_test - yhat_pt)^2 ~ 1))$coef[1:2]
r2_pt <- 1 - mse_pt[1] / var(y_test)

## 3. Bagging

fit_bagg <- randomForest(basico, ntree = 2000, nodesize = 20,mtry=11, data = data_train)
fit_bagg
yhat_bagg<-predict(fit_bagg,newdata = data_test)
mse_bagg <- summary(lm((y_test - yhat_bagg)^2 ~ 1))$coef[1:2]
r2_bagg<- 1 - mse_bagg[1] / var(y_test)

## 4. Random forest

fit_rf <- randomForest(basico,ntree = 2000, nodesize = 20, importance=TRUE,data = data_train)
fit_rf
yhat_rf<-predict(fit_rf,newdata = data_test)
mse_rf<- summary(lm((y_test - yhat_rf)^2 ~ 1))$coef[1:2]
r2_rf<- 1 - mse_rf[1] / var(y_test)


