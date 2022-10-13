# Databricks notebook source
##########################################
#########  Data Analysis French MTPL
#########  Install packages
#########  Author: Mario Wuthrich
#########  Version March 02, 2020
##########################################

##########################################
#########  install R packages
##########################################
install.packages("xts")
install.packages("sp")
install.packages("/dbfs/FileStore/tables/CASdatasets_1_0_12_tar.gz", repos = NULL, type ="source")
#install.packages("CASdatasets", repos = "http://dutangc.free.fr/pub/RRepos/", type="source")
require(MASS)
library(CASdatasets)
?CASdatasets
 
data(freMTPL2freq)
str(freMTPL2freq)

# COMMAND ----------

# MAGIC %sh sudo apt-get install libjpeg-turbo8-dev

# COMMAND ----------

install.packages("rms")

# COMMAND ----------

install.packages("jpeg")

# COMMAND ----------

#install.packages("jpeg",verbose= TRUE,dependencies = TRUE,repose = "https://cloud.r-project.org/")
library("jpeg")
install.packages("latticeExtra")
library("latticeExtra")
install.packages("Hmisc")

# COMMAND ----------

install.packages("rpart.plot")

# COMMAND ----------

##########################################
#########  Data Analysis French MTPL
#########  Benchmarks GLM
#########  Author: Mario Wuthrich
#########  Version March 02, 2020
##########################################

##########################################
#########  load packages and data
##########################################
 
source("./Tools/FreMTPL_1b load data.R")

# learning data
(l2 <- ddply(learn.GLM, .(ClaimNb), summarise, n=sum(n), exp=sum(Exposure)))
(lambda2<- sum(learn.GLM$ClaimNb)/sum(learn.GLM$Exposure))
round(100*l2$n/sum(l2$n),3) 
n_l

# test data
(l2 <- ddply(test.GLM, .(ClaimNb), summarise, n=sum(n), exp=sum(Exposure)))
sum(test.GLM$ClaimNb)/sum(test.GLM$Exposure)
round(100*l2$n/sum(l2$n),3) 
n_t

c(Poisson.Deviance(learn.GLM$Exposure*lambda2, learn.GLM$ClaimNb),Poisson.Deviance(test.GLM$Exposure*lambda2, test.GLM$ClaimNb))

# COMMAND ----------

##########################################
#########  GLM analysis
##########################################

str(learn.GLM)

### Model GLM1
{t1 <- proc.time()
  d.glm1 <- glm(ClaimNb ~ VehPowerGLM + VehAgeGLM + DrivAgeGLM + BonusMalusGLM
                       + VehBrand + VehGas + DensityGLM + Region + AreaGLM, 
                       data=learn.GLM, offset=log(Exposure), family=poisson())
(proc.time()-t1)[3]}
                   
summary(d.glm1)  
#anova(d.glm1)                                             
length(d.glm1$coefficients)

learn.GLM$fit <- fitted(d.glm1)
test.GLM$fit <- predict(d.glm1, newdata=test.GLM, type="response")
c(Poisson.Deviance(learn.GLM$fit, learn.GLM$ClaimNb),Poisson.Deviance(test.GLM$fit, test.GLM$ClaimNb))

# COMMAND ----------

### Model GLM2
{t1 <- proc.time()
d.glm2 <- glm(ClaimNb ~ VehPowerGLM + VehAgeGLM + DrivAgeGLM + BonusMalusGLM
                       + VehBrand + VehGas + DensityGLM + Region, 
                       data=learn.GLM, offset=log(Exposure), family=poisson())
(proc.time()-t1)[3]}
                   
summary(d.glm2)  
length(d.glm2$coefficients)


learn.GLM$fit <- fitted(d.glm2)
test.GLM$fit <- predict(d.glm2, newdata=test.GLM, type="response")
c(Poisson.Deviance(learn.GLM$fit, learn.GLM$ClaimNb),Poisson.Deviance(test.GLM$fit, test.GLM$ClaimNb))

# COMMAND ----------

### Model GLM3
{t1 <- proc.time()
d.glm3 <- glm(ClaimNb ~ VehPowerGLM + VehAgeGLM + DrivAgeGLM + BonusMalusGLM
                        + VehGas + DensityGLM + Region, 
                       data=learn.GLM, offset=log(Exposure), family=poisson())
(proc.time()-t1)[3]}
                   
summary(d.glm3)
#anova(d.glm3)  
length(d.glm3$coefficients)

learn.GLM$fit <- fitted(d.glm3)
test.GLM$fit <- predict(d.glm3, newdata=test.GLM, type="response")
c(Poisson.Deviance(learn.GLM$fit, learn.GLM$ClaimNb),Poisson.Deviance(test.GLM$fit, test.GLM$ClaimNb))
