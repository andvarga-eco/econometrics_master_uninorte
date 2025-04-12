library(glmnet)
library(dplyr)
library(hdm)
library(sandwich)
library(haven)
library(psych)

medexp<- read_dta("C:/Users/andre/OneDrive - Universidad del Norte/Drive/Uninorte/Uninorte-docencia/Econometría_MSC/2025/codigos/mus203mepsmedexp.dta")

# Objetivo: estimar el efecto del seguro suplementario sobre el gasto en salud

lapply(medexp, class)
medexp<-medexp%>%subset(!is.na(ltotexp))
describe(medexp$ltotexp) # Dependiente
describe(medexp$suppins) # Regresor de interés

medexp<-medexp%>%mutate(across(c(female,white,hisp,marry,northe,mwest,south,
                                   msa,phylim,actlim,injury,priolist,hvgg),~as.factor(.x)))

#OLS Básico

basico<-ltotexp~suppins+income+educyr+age+famsze+totchr+female+white+hisp+marry+northe+mwest+south+
  msa+phylim+actlim+injury+priolist+hvgg
full<-ltotexp~suppins+(income+educyr+age+famsze+totchr)^2+female+white+hisp+marry+northe+mwest+south+
  msa+phylim+actlim+injury+priolist+hvgg+(female+white+hisp+marry+northe+mwest+south+
                                            msa+phylim+actlim+injury+priolist+hvgg):(income+educyr+age+famsze+totchr)

regbasic<-lm(basico,data=medexp)
basic_coefs <- vcovHC(regbasic, type = "HC1")
basic_se<- sqrt(diag(basic_coefs))[2]
length(regbasic$coefficients)

regfull<-lm(full,data=medexp)
full_coefs <- vcovHC(regfull, type = "HC1")
full_se<- sqrt(diag(full_coefs))[2]
length(regfull$coefficients)

b_basic<-c(summary(regbasic)$coefficients[2,1],basic_se)
b_full<-c(summary(regfull)$coefficients[2,1],full_se)

# Lasso Partialling out

YW<-ltotexp~(income+educyr+age+famsze+totchr)^2+female+white+hisp+marry+northe+mwest+south+
  msa+phylim+actlim+injury+priolist+hvgg+(female+white+hisp+marry+northe+mwest+south+
                                            msa+phylim+actlim+injury+priolist+hvgg):(income+educyr+age+famsze+totchr)

DW<-suppins~(income+educyr+age+famsze+totchr)^2+female+white+hisp+marry+northe+mwest+south+
  msa+phylim+actlim+injury+priolist+hvgg+(female+white+hisp+marry+northe+mwest+south+
                                            msa+phylim+actlim+injury+priolist+hvgg):(income+educyr+age+famsze+totchr)

# partialling out W from Y
ywl <- rlasso(YW, data = medexp)
ywlres<-ywl$res
# Partialling out w from D
dwl <- rlasso(DW, data = medexp)
dwlres<-dwl$res
partial_lasso_fit <- lm(ywlres ~dwlres)
partial_lasso_est <- summary(partial_lasso_fit)$coef[2, 1]

hcv_coefs <- vcovHC(partial_lasso_fit, type = "HC1")
partial_lasso_se <- sqrt(diag(hcv_coefs))[2]

b_lasso<-c(partial_lasso_est,partial_lasso_se)


# Parialling out con CV
ywcv<-model.matrix(YW,medexp)
dwcv<-model.matrix(DW,medexp)

ywlcv<-cv.glmnet(ywcv,medexp$ltotexp,family = "gaussian", alpha = 1, nfolds = 5)
ltotexphat<-predict(ywlcv,newx=ywcv,s = "lambda.min")
ywlcvres<-medexp$ltotexp-ltotexphat

dwlcv<-cv.glmnet(dwcv,medexp$suppins,family = "gaussian", alpha = 1, nfolds = 5)
suppinshat<-predict(dwlcv,newx=dwcv,s = "lambda.min")
dwlcvres<-medexp$suppins-suppinshat


partial_lasso_cv<-lm(ywlcvres~dwlcvres)
plcv_est <- summary(partial_lasso_cv)$coef[2, 1]

plcv_coefs <- vcovHC(partial_lasso_cv, type = "HC1")
plcv_se <- sqrt(diag(plcv_coefs))[2]

b_lassocv<-c(plcv_est,plcv_se)

# Tabla
results<-cbind(b_basic,b_full,b_lasso,b_lassocv)
rownames(results)<-c("Coef","SE")
