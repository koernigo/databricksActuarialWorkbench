# Databricks notebook source
#Uncomment if Cluster doesn't install this package
install.packages("conflicted")

# COMMAND ----------

#Uncomment if Cluster doesn't install this package
install.packages("keras")

# COMMAND ----------

#Uncomment if Cluster doesn't install this package
install.packages("corrplot")

# COMMAND ----------

#Uncomment if Cluster doesn't install this package
install.packages("locfit")

# COMMAND ----------

## -----------------------------------------------------------------------------
library(conflicted)
library(SparkR)
library(keras)
library(locfit)
library(magrittr)
library(dplyr)
library(tibble)
library(purrr)
library(ggplot2)
library(gridExtra)
library(tidyr)
library(corrplot)
RNGversion("3.5.0")

# COMMAND ----------

conflict_prefer("ceiling", "base")
conflict_prefer("filter", "dplyr")
conflict_prefer("arrange", "dplyr")
conflict_prefer("mutate", "dplyr")
conflict_prefer("sample", "base")
conflict_prefer("summarize", "dplyr")
conflict_prefer("group_by", "dplyr")
conflict_prefer("n", "dplyr")
conflict_prefer("select", "dplyr")

# COMMAND ----------

## -----------------------------------------------------------------------------
options(encoding = 'UTF-8')


## -----------------------------------------------------------------------------
# set seed to obtain best reproducibility. note that the underlying architecture may affect results nonetheless, so full reproducibility cannot be guaranteed across different platforms.
seed <- 100
Sys.setenv(PYTHONHASHSEED = seed)
set.seed(seed)
reticulate::py_set_seed(seed)
tensorflow::tf$random$set_seed(seed)
