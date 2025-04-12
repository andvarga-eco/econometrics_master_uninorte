library(dplyr)
library(hdm)
library(glmnet)
library(sandwich)

data("GrowthData")

# Convergencia absoluta

abs<-lm(Outcome~gdpsh465,data=GrowthData)
b_abs<-summary(abs)$coefficients[2,1]
abs_coefs<-vcovHC(abs,type="HC1")
b_se_abs<-sqrt(diag(abs_coefs))[2]

b_absoluta<-c(b_abs,b_se_abs)

# Vectores de variables

y<-as.vector(GrowthData$Outcome)
D<-as.vector(GrowthData$gdpsh465)
controls<-as.matrix(GrowthData)[,-c(1,2,3)]

# Convergencia condicional

full<-y~D+controls
fullreg<-lm(full,data=GrowthData)
b_est_full<-summary(fullreg)$coefficients[2,1]
full_coefs<-vcovHC(fullreg,type="HC1")
b_se_full<-sqrt(diag(full_coefs))[2]

b_c_ols<-c(b_est_full,b_se_full)

# Doble Lasso: lambda teórico
yc<-rlasso(y~controls,data=GrowthData,post=FALSE)
ycres<-yc$res

Dc<-rlasso(D~controls,data=GrowthData,post=FALSE)
Dcres<-Dc$res

partialdl<-lm(ycres~Dcres)
b_est_dl<-summary(partialdl)$coefficients[2,1]
dl_coefs<-vcovHC(partialdl,type="HC1")
b_se_dl<-sqrt(diag(dl_coefs))[2]

b_c_dl<-c(b_est_dl,b_se_dl)

# Doble Post-Lasso (PDS)
## manual
ycp<-rlasso(y~controls,data=GrowthData,post=TRUE)
ycpres<-ycp$res

Dcp<-rlasso(D~controls,data=GrowthData,post=TRUE)
Dcpres<-Dcp$res

partialdpl<-lm(ycpres~Dcpres)
b_est_dpl<-summary(partialdpl)$coefficients[2,1]
dpl_coefs<-vcovHC(partialdpl,type="HC1")
b_se_dpl<-sqrt(diag(dpl_coefs))[2]

b_c_dpl<-c(b_est_dpl,b_se_dpl)

## Dos en uno
pds<-rlassoEffect(controls,y,D,method="partialling out")
summary(pds)$coeff[,1:2]

# Doble Lasso CV

yccv<-cv.glmnet(controls,y,alpha=1,nfolds=5)
yhat<-predict(yccv,newx=controls,s="lambda.min")
yccvres<-y-yhat

Dccv<-cv.glmnet(controls,D,alpha=1,nfolds=5)
Dhat<-predict(Dccv,newx=controls,s="lambda.min")
Dccvres<-D-Dhat

pcv<-lm(yccvres~Dccvres)
b_est_cv<-summary(pcv)$coefficients[2,1]
dcv_coefs<-vcovHC(pcv,type="HC1")
b_se_cv<-sqrt(diag(dcv_coefs))[2]

b_c_cv<-c(b_est_cv,b_se_cv)

results<-cbind(b_absoluta,b_c_ols,b_c_dl,b_c_cv,b_c_dpl)
rownames(results)<-c("Coef","SE")
