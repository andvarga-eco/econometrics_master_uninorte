
# Objetivo: desempeño de OLS a medida que aumenta la dimensionalidad

## Librerias

library(dplyr)
library(purrr)

## Cargar los datos

file <- "https://raw.githubusercontent.com/CausalAIBook/MetricsMLNotebooks/main/data/wage2015_subsample_inference.csv"
data <- read.csv(file)
dim(data)

## Media muestral

desc<-data%>%summarise(across(lwage:exp1,~mean(.x)))
colnames(desc) <- c("Log Wage", "Female", "Some High School",
                     "High School Graduate", "Some College", "College Graduate",
                     "Advanced Degree", "Midwest", "South", "West", "Northeast", "Experience")
rownames(desc)<-c("Sample mean")

## Dividir la muestra: train-test

data$occ2<-factor(data$occ2) # definir como factor
data$ind2<-factor(data$ind2) # definir como factor

set.seed(1) # to make the results replicable (we will generate random numbers)
n <- length(data$lwage)
random <- sample(1:n, floor(n * 4 / 5)) # draw (4/5)*n random numbers from 1 to n without replacing
train <- data[random, ]
test <- data[-random, ]

## Modelos a estimar

basico<-lwage ~ (sex + exp1 + shs + hsg + scl + clg + mw + so + we + occ2 + ind2)

flexible<-lwage ~ sex + shs + hsg + scl + clg + mw + so + we + occ2 + ind2 +
  (exp1 + exp2 + exp3 + exp4) * (shs + hsg + scl + clg + occ2 + ind2 + mw + so + we)

extraflex <- lwage ~ sex + (exp1 + exp2 + exp3 + exp4 + shs + hsg + scl + clg + occ2 +
                              ind2 + mw + so + we)^2


## Estimación de modelos dentro de muestra

regbasic<-lm(basico,data=train)
length(regbasic$coefficients) # número de coeficientes estimados
r2_b_s<-(summary(regbasic))$r.squared
mse_b_s<-mean((regbasic$residuals)^2)
  
regflex<-lm(flexible,data=train)
length(regflex$coefficients) # número de coeficientes estimados
r2_f_s<-(summary(regflex))$r.squared
mse_f_s<-mean((regflex$residuals)^2)

regxflex<-lm(extraflex,data=train)
length(regxflex$coefficients) # número de coeficientes estimados
r2_fx_s<-(summary(regxflex))$r.squared
mse_fx_s<-mean((regxflex$residuals)^2)


## Predicción de modelos fuera de muestra
ytest<-test$lwage
yhat_basic<-predict(regbasic,newdata = test)
mse_b_p<-mean((ytest-yhat_basic)^2)
r2_b_p<-1-mse_b_p/(mean((ytest-mean(ytest))^2))

yhat_flex<-predict(regflex,newdata = test)
mse_f_p<-mean((ytest-yhat_flex)^2)
r2_f_p<-1-mse_f_p/(mean((ytest-mean(ytest))^2))

yhat_xflex<-predict(regxflex,newdata = test)
mse_fx_p<-mean((ytest-yhat_xflex)^2)
r2_fx_p<-1-mse_fx_p/(mean((ytest-mean(ytest))^2))

## MSE y R2 dentro y fuera de muestra

results<-c(round(mse_b_s,2),round(mse_f_s,2),round(mse_fx_s,2),
           round(r2_b_s,2),round(r2_f_s,2),round(r2_fx_s,2),
           round(mse_b_p,2),round(mse_f_p,2),round(mse_fx_p,2),
           round(r2_b_p,2),round(r2_f_p,2),round(r2_fx_p,2))
rtable<-matrix(results,3,4)
colnames(rtable)<-c("MSE_s","R2_s","MSE_p","R2_p")
rownames(rtable)<-c("Básico","Flexible","Extraflexible")



