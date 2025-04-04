# Lasso para predicción wage

library(glmnet)
library(dplyr)
library(hdm)

## Predicción wage

file <- "https://raw.githubusercontent.com/CausalAIBook/MetricsMLNotebooks/main/data/wage2015_subsample_inference.csv"
data <- read.csv(file)
data<-data%>%mutate(sex=factor(sex),
                    occ2=factor(occ2),
                    ind2=factor(ind2))
n<-length(data$lwage)
set.seed(1) # to make the results replicable (we will generate random numbers)
random <- sample(1:n, floor( n* 4 / 5)) # draw (4/5)*n random numbers from 1 to n without replacing
train <- data[random, ]
test <- data[-random, ]

### Modelo extraflexible: data split

extraflex <- lwage ~ sex + (exp1 + exp2 + exp3 + exp4 + shs + hsg + scl + clg + occ2 + ind2 + mw + so + we)^2


flex_data <- model.matrix(extraflex, data)
train_flex <- flex_data[random, ]
test_flex <- flex_data[-random, ]

fit_lcv <- cv.glmnet(train_flex, train$lwage, family = "gaussian", alpha = 1, nfolds = 5)
print(fit_lcv)
plot(fit_lcv)
fit_lcv$lambda.min
yhat_lcv <- predict(fit_lcv, newx = train_flex, s = "lambda.min")

r2_l <- 1 - sum((yhat_lcv - train$lwage)^2) / sum((train$lwage - mean(train$lwage))^2)
lasso_res <- train$lwage - yhat_lcv
mse_l <- mean(lasso_res^2)

yhat_lcv_test <- predict(fit_lcv, newx = test_flex, s = "lambda.min")
mse_lasso <- sum((test$lwage - yhat_lcv_test)^2) / length(test$lwage)
r2_lasso <- 1 - mse_lasso / mean((test$lwage - mean(test$lwage))^2)

## POstLasso

fit_rlasso_post_extra <- rlasso(formula_extra, data_train, post = TRUE)
yhat_rlasso_post_extra <- predict(fit_rlasso_post_extra, newdata = data_test)
mse_lasso_post_extra <- summary(lm((y_test - yhat_rlasso_post_extra)^2 ~ 1))$coef[1:2]
r2_lasso_post_extra <- 1 - mse_lasso_post_extra[1] / var(y_test)



