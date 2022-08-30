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


