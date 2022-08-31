# Databricks notebook source
# MAGIC %md # MLflow quickstart: tracking
# MAGIC 
# MAGIC This notebook creates a Random Forest model on a simple dataset and uses the MLflow Tracking API to log the model, selected model parameters and evaluation metrics, and other artifacts.
# MAGIC 
# MAGIC ## Requirements
# MAGIC * This notebook requires Databricks Runtime 6.4 or above, or Databricks Runtime 6.4 ML or above.

# COMMAND ----------

# MAGIC %md This notebook does not use distributed processing, so you can use the R `install.packages()` function to install packages on the driver node only.  
# MAGIC To take advantage of distributed processing, you must install packages on all nodes in the cluster by creating a cluster library. See "Install a library on a cluster" ([AWS](https://docs.databricks.com/libraries/cluster-libraries.html#install-a-library-on-a-cluster)|[Azure](https://docs.microsoft.com/en-us/azure/databricks/libraries/cluster-libraries#--install-a-library-on-a-cluster)|[GCP](https://docs.gcp.databricks.com/libraries/cluster-libraries.html#install-a-library-on-a-cluster)).

# COMMAND ----------

install.packages("mlflow")

# COMMAND ----------

library(mlflow)
install_mlflow()

# COMMAND ----------

# MAGIC %md Import the required libraries.  
# MAGIC 
# MAGIC This notebook uses the R library `carrier` to serialize the predict method of the trained model, so that it can be loaded back into memory later. For more information, see the [`carrier` github repo](https://github.com/r-lib/carrier). 

# COMMAND ----------

install.packages("carrier")
install.packages("e1071")

library(MASS)
library(caret)
library(e1071)
library(randomForest)
library(SparkR)
library(carrier)

# COMMAND ----------

with(mlflow_start_run(), {
  
  # Set the model parameters
  ntree <- 100
  mtry <- 3
  
  # Create and train model
  rf <- randomForest(type ~ ., data=Pima.tr, ntree=ntree, mtry=mtry)
  
  # Use the model to make predictions on the test dataset
  pred <- predict(rf, newdata=Pima.te[,1:7])
  
  # Log the model parameters used for this run
  mlflow_log_param("ntree", ntree)
  mlflow_log_param("mtry", mtry)
  
  # Define metrics to evaluate the model
  cm <- confusionMatrix(pred, reference = Pima.te[,8])
  sensitivity <- cm[["byClass"]]["Sensitivity"]
  specificity <- cm[["byClass"]]["Specificity"]
  
  # Log the value of the metrics 
  mlflow_log_metric("sensitivity", sensitivity)
  mlflow_log_metric("specificity", specificity)
  
  # Log the model
  # The crate() function from the R package "carrier" stores the model as a function
  predictor <- crate(function(x) predict(rf,.x))
  mlflow_log_model(predictor, "model")     
  
  # Create and plot confusion matrix
  png(filename="confusion_matrix_plot.png")
  barplot(as.matrix(cm), main="Results",
         xlab="Observed", ylim=c(0,200), col=c("green","blue"),
         legend=rownames(cm), beside=TRUE)
  dev.off()
  
  # Save the plot and log it as an artifact
  mlflow_log_artifact("confusion_matrix_plot.png") 
    
})

# COMMAND ----------

# MAGIC %md To view the results, click **Experiment** at the upper right of this page. The Experiments sidebar appears. This sidebar displays the parameters and metrics for each run of this notebook. Click the circular arrows icon to refresh the display to include the latest runs. 
# MAGIC 
# MAGIC When you click the square icon with the arrow to the right of the date and time of the run, the Runs page opens in a new tab. This page shows all of the information that was logged from the run. Scroll down to the Artifacts section to find the logged model and plot.
# MAGIC 
# MAGIC For more information, see "View notebook experiment" ([AWS](https://docs.databricks.com/applications/mlflow/tracking.html#view-notebook-experiment)|[Azure](https://docs.microsoft.com/azure/databricks/applications/mlflow/tracking#view-notebook-experiment)|[GCP](https://docs.gcp.databricks.com/applications/mlflow/tracking.html#view-notebook-experiment)).
