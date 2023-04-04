# Databricks notebook source
# MAGIC %sql
# MAGIC drop database if exists actuarial_db CASCADE

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE DATABASE IF NOT EXISTS actuarial_db

# COMMAND ----------

# MAGIC %sql 
# MAGIC use actuarial_db

# COMMAND ----------

# MAGIC %sql
# MAGIC drop table if exists workerscomp_bronze

# COMMAND ----------

load("./_setup/WorkersComp.RData")

# COMMAND ----------

library(SparkR)

# COMMAND ----------

sparkR.session()
write_format = "delta"
dataset="workerscomp_bronze"
dataset_descr = "Synthetic Workers Comp Claims"
save_path = paste("/insurance/delta/WC/",dataset,sep="")
table_name = dataset
# Write the data to its target.
data_set_df <- createDataFrame( as.data.frame(WorkersComp))
write.df(data_set_df, source = write_format, path = save_path,mode = "OVERWRITE")
# Create the table.
command = paste("CREATE TABLE actuarial_db.", table_name, " USING DELTA LOCATION '", save_path, "'", " COMMENT '",dataset_descr,"'",sep = "")
print(command)
result <- SparkR::sql(command)
print(result)
