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

# modelos posibles
m1<-y~V1
m2<-y~V2
m3<-y~V3
m4<-y~V1+V2
m5<-y~V1+V3
m6<-y~V2+V3
m7<-y~V1+V2+V3
models<-c(m1,m2,m3,m4,m5,m6,m7)

# Single-split

set.seed(10101) # to make the results replicable (we will generate random numbers)
random <- sample(1:length(y), floor(n * 4 / 5)) # draw (4/5)*n random numbers from 1 to n without replacing
train <- data[random, ]
test <- data[-random, ]

## MSE para todos los modelos

for (model in models){
  m<-lm(model,data=data.frame(train))
  y_test<-predict(m,newdata=data.frame(test))
  print(sum((test[,4] - y_test)^2) / length(y_test))
  }

# K-fold cross validation: full model

library(caret)
set.seed(10101)
train_control<-trainControl(method="cv", number=5)
mcv<-train(y~V1+V2+V3,data=data.frame(data),
           method="lm", trControl=train_control)
print(mcv)
summary(mcv)

# K-fold cross validation: todos los modelos

library(caret)
for (model in models){
set.seed(10101)
train_control<-trainControl(method="cv", number=5)
mcv<-train(model,data=data.frame(data),
           method="lm", trControl=train_control)
print(mcv)
}


