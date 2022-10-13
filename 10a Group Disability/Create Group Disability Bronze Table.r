# Databricks notebook source
# MAGIC %sql
# MAGIC drop database ins_data_sets CASCADE

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE DATABASE IF NOT EXISTS ins_data_sets

# COMMAND ----------

# MAGIC %sql 
# MAGIC use ins_data_sets

# COMMAND ----------

# MAGIC %sql
# MAGIC drop table if exists workerscomp_bronze

# COMMAND ----------

load("WorkersComp.RData")

# COMMAND ----------

library(SparkR)

# COMMAND ----------

sparkR.session()
write_format = "delta"
dataset="workerscomp_bronze"
dataset_descr = "Synthetic Workers Comp Claims"
save_path = paste("/tmp/delta/WC/",dataset,sep="")
table_name = dataset
# Write the data to its target.
data_set_df <- createDataFrame( as.data.frame(WorkersComp))
write.df(data_set_df, source = write_format, path = save_path)
# Create the table.
command = paste("CREATE TABLE ins_data_sets.", table_name, " USING DELTA LOCATION '", save_path, "'", " COMMENT '",dataset_descr,"'",sep = "")
print(command)
result <- SparkR::sql(command)
print(result)

# COMMAND ----------

# MAGIC %sql
# MAGIC show table extended like 'workerscomp_bronze'
