
# Objetivo: desempeño predictivo OLS cuando p/n no es pequeño

library(dplyr)
library(glmnet)
library(hdm)
library(sandwich)
library(lmtest)

file <- "https://raw.githubusercontent.com/CausalAIBook/MetricsMLNotebooks/main/data/wage2015_subsample_inference.csv"
data <- read.csv(file)

data<-data%>%mutate(sex=factor(sex),
                    occ2=factor(occ2),
                    ind2=factor(ind2))

mean_desc<-data%>%group_by(sex)%>%summarise(across(lwage:ne,~mean(.x,na.rm=TRUE)))
mean_desc_full<-data%>%summarise(across(lwage:ne,~mean(.x,na.rm=TRUE)))

# Modelo sin controles

nocontrol<-lm(lwage~sex,data=data)
coeftest(nocontrol,vcov=vcovHC(nocontrol, type = "HC3"))

# Modelo con matriz de controles W: raw + transformaciones + interacciones

flex<-lwage~sex+(exp1+exp2+exp3+exp4)*(shs+hsg + scl + clg + occ2 + ind2 + mw + so + we)

control<-lm(flex,data=data)
control_est <- summary(control)$coef[2, 1]
summary(control)
hcv_coefs <- vcovHC(control, type = "HC3")
control_se <- sqrt(diag(hcv_coefs))[2] # Estimated std errors
print(control_est)
print(control_se)

# Modelo extra flexible

extraflex <- lwage ~ sex + (exp1 + exp2 + exp3 + exp4 + shs + hsg + scl + clg + occ2 + ind2 + mw + so + we)^2

## Sobre la muestra completa

control_extra<-lm(extraflex,data=data)
control_extra_est <- summary(control_extra)$coef[2, 1]
hcv_coefs_extra <- vcovHC(control_extra, type = "HC0")
control_extra_se <- sqrt(diag(hcv_coefs_extra))[2] # Estimated std errors
print(control_extra_est)
print(control_extra_se)
print(summary(control_extra)$r.squared)
print(summary(control_extra)$adj.r.squared)
print(mean(control_extra$residuals^2))

## Sobre n=1000

set.seed(2724)
subset_size <- 1000
random <- sample(seq_len(nrow(data)), subset_size)
subset <- data[random, ]

control_extra_s<-lm(extraflex,data=subset)
control_extra_est_s <- summary(control_extra_s)$coef[2, 1]
hcv_coefs_extra_s <- vcovHC(control_extra_s, type = "HC0")
control_extra_se_s <- sqrt(diag(hcv_coefs_extra_s))[2] # Estimated std errors
print(control_extra_est_s)
print(control_extra_se_s)
print(summary(control_extra_s)$r.squared)
print(summary(control_extra_s)$adj.r.squared)
print(mean(control_extra_s$residuals^2))
