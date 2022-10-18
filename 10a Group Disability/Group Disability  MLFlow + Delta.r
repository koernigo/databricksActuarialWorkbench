# Databricks notebook source
# DBTITLE 1,An End To End Actuarial Workflow for Group Disability Claim Size Modeling
# MAGIC %md
# MAGIC ![my_test_image](files/WC_E2E.png)

# COMMAND ----------

# DBTITLE 1,Library Pre-Requisites
#Uncomment if Cluster doesn't install this package
install.packages("conflicted")

# COMMAND ----------

#Uncomment if Cluster doesn't install this package
install.packages("keras")

# COMMAND ----------

#Uncomment if Cluster doesn't install this package
install.packages("locfit")

# COMMAND ----------

#Uncomment if Cluster doesn't install this package
install.packages("corrplot")

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

# COMMAND ----------

# DBTITLE 1,Helper Functions
## -----------------------------------------------------------------------------
# https://stackoverflow.com/questions/65366442/cannot-convert-a-symbolic-keras-input-output-to-a-numpy-array-typeerror-when-usi
# https://tensorflow.rstudio.com/guide/tfhub/examples/feature_column/
tensorflow::tf$compat$v1$disable_eager_execution()


## -----------------------------------------------------------------------------
ax_limit <- c(0,50000)
line_size <- 1.1


## -----------------------------------------------------------------------------
# MinMax scaler
preprocess_minmax <- function(varData) {
  X <- as.numeric(varData)
  2 * (X - min(X)) / (max(X) - min(X)) - 1
}


## -----------------------------------------------------------------------------
# One Hot encoding for categorical features
preprocess_cat_onehot <- function(data, varName, prefix) {
  varData <- data[[varName]]
  X <- as.integer(varData)
  n0 <- length(unique(X))
  n1 <- 1:n0
  addCols <- purrr::map(n1, function(x, y) {as.integer(y == x)}, y = X) %>%
    rlang::set_names(paste0(prefix, n1))
  cbind(data, addCols)
}


## -----------------------------------------------------------------------------
#https://stat.ethz.ch/pipermail/r-help/2013-July/356936.html
scale_no_attr <- function (x, center = TRUE, scale = TRUE) 
{
    x <- as.matrix(x)
    nc <- ncol(x)
    if (is.logical(center)) {
        if (center) {
            center <- colMeans(x, na.rm = TRUE)
            x <- sweep(x, 2L, center, check.margin = FALSE)
        }
    }
    else if (is.numeric(center) && (length(center) == nc)) 
        x <- sweep(x, 2L, center, check.margin = FALSE)
    else stop("length of 'center' must equal the number of columns of 'x'")
    if (is.logical(scale)) {
        if (scale) {
            f <- function(v) {
                v <- v[!is.na(v)]
                sqrt(sum(v^2)/max(1, length(v) - 1L))
            }
            scale <- apply(x, 2L, f)
            x <- sweep(x, 2L, scale, "/", check.margin = FALSE)
        }
    }
    else if (is.numeric(scale) && length(scale) == nc) 
        x <- sweep(x, 2L, scale, "/", check.margin = FALSE)
    else stop("length of 'scale' must equal the number of columns of 'x'")
    #if (is.numeric(center)) 
    #    attr(x, "scaled:center") <- center
    #if (is.numeric(scale)) 
    #    attr(x, "scaled:scale") <- scale
    x
}

# COMMAND ----------

## -----------------------------------------------------------------------------
square_loss <- function(y_true, y_pred){mean((y_true-y_pred)^2)}
gamma_loss  <- function(y_true, y_pred){2*mean((y_true-y_pred)/y_pred + log(y_pred/y_true))}
ig_loss     <- function(y_true, y_pred){mean((y_true-y_pred)^2/(y_pred^2*y_true))}
p_loss      <- function(y_true, y_pred, p){2*mean(y_true^(2-p)/((1-p)*(2-p))-y_true*y_pred^(1-p)/(1-p)+y_pred^(2-p)/(2-p))}

k_gamma_loss  <- function(y_true, y_pred){2*k_mean(y_true/y_pred - 1 - log(y_true/y_pred))}
k_ig_loss     <- function(y_true, y_pred){k_mean((y_true-y_pred)^2/(y_pred^2*y_true))}
k_p_loss      <- function(y_true, y_pred){2*k_mean(y_true^(2-p)/((1-p)*(2-p))-y_true*y_pred^(1-p)/(1-p)+y_pred^(2-p)/(2-p))}


## -----------------------------------------------------------------------------
keras_plot_loss_min <- function(x, seed) {
    x <- x[[2]]
    ylim <- range(x)
    vmin <- which.min(x$val_loss)
    df_val <- data.frame(epoch = 1:length(x$loss), train_loss = x$loss, val_loss = x$val_loss)
    df_val <- gather(df_val, variable, loss, -epoch)
    #Added for mlFlow tracking
    plt <- ggplot(df_val, aes(x = epoch, y = loss, group = variable, color = variable)) +
      geom_line(size = line_size) + geom_vline(xintercept = vmin, color = "green", size = line_size) +
      labs(title = paste("Train and validation loss for seed", seed),
           subtitle = paste("Green line: Smallest validation loss for epoch", vmin))
    ggsave("/dbfs/tmp/keras_plot_loss.png")
    suppressMessages(print(plt))
}

# COMMAND ----------

## -----------------------------------------------------------------------------
plot_size <- function(test, xvar, title, model, mdlvariant) {
  out <- test %>% group_by(!!sym(xvar)) %>%
    summarize(obs = mean(Claim) , pred = mean(!!sym(mdlvariant)))
  
  ggplot(out, aes(x = !!sym(xvar), group = 1)) +
    geom_point(aes(y = pred, colour = model)) +
    geom_point(aes(y = obs, colour = "observed")) +
    geom_line(aes(y = pred, colour = model), linetype = "dashed") +
    geom_line(aes(y = obs, colour = "observed"), linetype = "dashed") +
    ylim(ax_limit) + labs(x = xvar, y = "claim size", title = title) +
    theme(legend.position = "bottom")
}

# COMMAND ----------

# DBTITLE 1,General data preprocessing
# MAGIC %md
# MAGIC We drop 10 claims with claim occurrence year 1987. This provides us with 99'990 claims, all having occurred between 1988/01/01 and 2006/07/20. These claims have been reported in the calendar years from 1998 to 2008. The maximal reporting delay is 1'042 days, this corresponds to 2.85 years or to almost 149 weeks. We drop all claims with non-positive HoursWorkedPerWeek.

# COMMAND ----------

# DBTITLE 1,Load Workers Comp Data From Bronze Table
GroupDisability_Spark_DF<-SparkR::sql("SELECT * FROM ins_data_sets.groupdisability_bronze")
GroupDisability <- as.data.frame(GroupDisability_Spark_DF)
GroupDisability <- mutate_at(GroupDisability, vars("MaritalStatus", "Gender","PartTimeFullTime"), as.factor)

# COMMAND ----------

str(GroupDisability)

# COMMAND ----------

## -----------------------------------------------------------------------------
#load(file.path("GroupDisability.RData"))  # relative path to .Rmd file

# COMMAND ----------

## -----------------------------------------------------------------------------
dat <- GroupDisability %>% filter(AccYear > 1987, HoursWorkedPerWeek > 0)

# COMMAND ----------

## -----------------------------------------------------------------------------
# Order claims in decreasing order for split train/test (see below), and add an ID
dat <- dat %>% arrange(desc(Claim))
dat <- dat %>% mutate(Id=1:nrow(dat))

# COMMAND ----------

str(dat)

# COMMAND ----------

# DBTITLE 1,Store Silver Table To Delta Lake
write_format = "delta"
dataset="groupdisability_silver"
dataset_descr = "Group Disability Claims Silver"
save_path = paste("/tmp/delta/GD/",dataset,sep="")
dbutils.fs.rm(save_path,"true")
table_name = dataset
# Write the data to its target.
data_set_df <- createDataFrame(dat)
write.df(data_set_df, source = write_format, path = save_path)
# Create the table.
command1 = paste("DROP TABLE IF EXISTS ins_data_sets.", table_name)
command2 = paste("CREATE TABLE ins_data_sets.", table_name, " USING DELTA LOCATION '", save_path, "'", " COMMENT '",dataset_descr,"'",sep = "")
#print(command2)
result <- SparkR::sql(command1)
print(result)
result <- SparkR::sql(command2)
print(result)

# COMMAND ----------

# DBTITLE 1,OPTIONAL: Use Databricks Visualizations
# MAGIC %python
# MAGIC df = spark.read.load('/tmp/delta/GD/groupdisability_silver')
# MAGIC display(df)

# COMMAND ----------

## -----------------------------------------------------------------------------
# scaling and cut-off
dat <- dat %>% mutate(
        Age = pmax(16, pmin(70, Age)),
        AgeNN = scale_no_attr(Age),
        GenderNN = as.integer(Gender),
        GenderNN = scale_no_attr(GenderNN),
        DependentChildren = pmin(1, DependentChildren),
        DependentChildrenNN = scale_no_attr(DependentChildren),
        DependentsOther = pmin(1, DependentsOther),
        DependentsOtherNN = scale_no_attr(DependentsOther),
        WeeklyPay = pmin(1200, WeeklyPay),
        WeeklyPayNN = scale_no_attr(WeeklyPay),
        PartTimeFullTimeNN = scale_no_attr(as.integer(PartTimeFullTime)),
        HoursWorkedPerWeek = pmin(60, HoursWorkedPerWeek),
        HoursWorkedPerWeekNN = scale_no_attr(HoursWorkedPerWeek),
        DaysWorkedPerWeekNN = scale_no_attr(DaysWorkedPerWeek),
        AccYearNN = scale_no_attr(AccYear),
        AccMonthNN = scale_no_attr(AccMonth),
        AccWeekdayNN = scale_no_attr(AccWeekday),
        AccTimeNN = scale_no_attr(AccTime),
        RepDelay = pmin(100, RepDelay),
        RepDelayNN = scale_no_attr(RepDelay)
)


## -----------------------------------------------------------------------------
# one-hot encoding (not dummy encoding!)
dat <- dat %>% preprocess_cat_onehot("MaritalStatus", "Marital")


## -----------------------------------------------------------------------------
# add two additional randomly generated features (later used)
set.seed(seed)

dat <- dat %>% mutate(
    RandNN = rnorm(nrow(dat)),
    RandNN = scale_no_attr(RandNN),
    RandUN = runif(nrow(dat), min = -sqrt(3), max = sqrt(3)),
    RandUN = scale_no_attr(RandUN)
)

# COMMAND ----------

## -----------------------------------------------------------------------------
head(dat)

# COMMAND ----------

## -----------------------------------------------------------------------------
str(dat)

# COMMAND ----------

colnames(dat)

# COMMAND ----------

## -----------------------------------------------------------------------------
summary(dat)

# COMMAND ----------

# DBTITLE 1,Split train and test data
# MAGIC %md

# COMMAND ----------

## -----------------------------------------------------------------------------
idx <- sample(x = c(1:5), size = ceiling(nrow(dat) / 5), replace = TRUE)
idx <- (1:ceiling(nrow(dat) / 5) - 1) * 5 + idx

test <- dat[intersect(idx, 1:nrow(dat)), ]
learn <- dat[setdiff(1:nrow(dat), idx), ]

learn <- learn[sample(1:nrow(learn)), ]
test <- test[sample(1:nrow(test)), ]


## -----------------------------------------------------------------------------
# size of train/test
sprintf("Number of observations (learn): %s", nrow(learn))
sprintf("Number of observations (test): %s", nrow(test))


## -----------------------------------------------------------------------------
# Claims average of learn/test
sprintf("Empirical claims average (learn): %s", round(sum(learn$Claim) / length(learn$Claim), 0))
sprintf("Empirical claims average (test): %s", round(sum(test$Claim) / length(test$Claim), 0))


## -----------------------------------------------------------------------------
# Quantiles of learn/test
probs <- c(.1, .25, .5, .75, .9)
bind_rows(quantile(learn$Claim, probs = probs), quantile(test$Claim, probs = probs))

# COMMAND ----------

# DBTITLE 1,Reporting Delay summary statistics
## -----------------------------------------------------------------------------
range(dat$AccYear)

# COMMAND ----------

## -----------------------------------------------------------------------------
range(dat$RepYear)

# COMMAND ----------

## -----------------------------------------------------------------------------
range(dat$RepDelay)

# COMMAND ----------

## -----------------------------------------------------------------------------
round(range(dat$RepDelay) / 365, 4)

# COMMAND ----------

## -----------------------------------------------------------------------------
round(range(dat$RepDelay) / 7, 4)

# COMMAND ----------

## -----------------------------------------------------------------------------
round(mean(dat$RepDelay) / 7, 4)

# COMMAND ----------

## -----------------------------------------------------------------------------
# define acc_week, rep_week and RepDelay_week
min_accDay <- min(dat$AccDay)
dat <- dat %>% mutate(
    acc_week = floor((AccDay - min_accDay) / 7),
    rep_week = floor((RepDay - min_accDay) / 7),
    RepDelay_week = rep_week - acc_week
)

# COMMAND ----------

## -----------------------------------------------------------------------------
quantile(dat$RepDelay_week, probs = c(.9, .98, .99))

# COMMAND ----------

## -----------------------------------------------------------------------------
acc1 <- dat %>% group_by(acc_week, rep_week) %>% summarize(nr = n())
acc2 <- dat %>% group_by(acc_week) %>% summarize(mm = mean(RepDelay_week))

# COMMAND ----------

## -----------------------------------------------------------------------------
head(acc1)

# COMMAND ----------

## -----------------------------------------------------------------------------
head(acc2)

# COMMAND ----------

## -----------------------------------------------------------------------------
# to plot the quantiles
qq0 <- c(.9, .975, .99)
qq1 <- quantile(dat$RepDelay_week, probs = qq0)
qq1

# COMMAND ----------

## -----------------------------------------------------------------------------
ggplot(acc1, aes(x = rep_week - acc_week, y = max(acc_week) - acc_week)) +
    geom_point() +
    geom_point(data = acc2, aes(x = mm, y = acc_week), color = "cyan") +
    geom_vline(xintercept = qq1, color = "orange", size = line_size) +
    scale_y_continuous(
      labels = rev(c(min(dat$AccYear):(max(dat$AccYear) + 1))),
      breaks = c(0:(max(dat$AccYear) - min(dat$AccYear) + 1)) * 365 / 7 - 25
    ) + labs(title = "Claims reporting", x = "reporting delay (in weeks)", y = "accident date")

# COMMAND ----------

# DBTITLE 1,Claim size summary statistic
## -----------------------------------------------------------------------------
range(dat$Claim)

# COMMAND ----------

## -----------------------------------------------------------------------------
summary(dat$Claim)

# COMMAND ----------

# DBTITLE 1,Plotting
## -----------------------------------------------------------------------------
p1 <- ggplot(dat %>% filter(Claim <= 10000), aes(x = Claim)) + geom_density(colour = "blue") +
    labs(title = "Empirical density of claims amounts", x = "claims amounts (<=10000)", y = "empirical density")

p2 <- ggplot(dat, aes(x = log(Claim))) + geom_density(colour = "blue") +
    labs(title = "Empirical density of log(claims amounts)", x = "claims amounts", y = "empirical density")

p3 <- ggplot(dat, aes(x = Claim^(1/3))) + geom_density(colour = "blue") +
    labs(title = "Empirical density of claims amounts^(1/3)", x = "claims amounts", y = "empirical density")

grid.arrange(p1, p2, p3, ncol = 1)

# COMMAND ----------

# DBTITLE 1,Log-log plot
## -----------------------------------------------------------------------------
pp <- ecdf(dat$Claim)
xx <- min(log(dat$Claim)) + 0:100/100 * (max(log(dat$Claim)) - min(log(dat$Claim)))
ES <- predict(locfit(log(1.00001 - pp(dat$Claim)) ~ log(dat$Claim), alpha = 0.1, deg = 2), newdata = xx)
dat_loglog <- data.frame(xx = xx, ES = ES)


## -----------------------------------------------------------------------------
ggplot(dat_loglog, aes(x = xx, y = ES)) + geom_line(colour = "blue", size = line_size) +
    geom_vline(xintercept = log(c(1:10) * 1000), colour = "green", linetype = "dashed") +
    geom_vline(xintercept = log(c(1:10) * 10000), colour = "yellow", linetype = "dashed") +
    geom_vline(xintercept = log(c(1:10) * 100000), colour = "orange", linetype = "dashed") +
    geom_vline(xintercept = log(c(1:10) * 1000000), colour = "red", linetype = "dashed") +
    labs(title = "log-log plot of claim amounts", x = "logged claim amount", y = "logged survival probability") +
    scale_x_continuous(breaks = seq(3,16,1))

# COMMAND ----------

## -----------------------------------------------------------------------------
col_names <- c("Age","Gender","MaritalStatus","DependentChildren","DependentsOther",
               "WeeklyPay","PartTimeFullTime","HoursWorkedPerWeek","DaysWorkedPerWeek",
               "AccYear","AccMonth","AccWeekday","AccTime","RepDelay")

global_avg <- mean(dat$Claim)
severity_limits <- c(0,40000)


## -----------------------------------------------------------------------------
dat_tmp <- dat
dat_tmp <- dat_tmp %>% mutate(
    WeeklyPay = pmin(1200, ceiling(WeeklyPay / 100) * 100),
    HoursWorkedPerWeek = pmin(60, HoursWorkedPerWeek),
    RepDelay = pmin(100, floor(RepDelay / 10) * 10)
)

# COMMAND ----------

# DBTITLE 1,Marginal plots
for (k in 1:length(col_names)) {
    xvar <- col_names[k]
    out <- dat_tmp %>% group_by(!!sym(xvar)) %>% summarize(vol = n(), avg = mean(Claim))

    tmp <- dat_tmp %>% select(!!sym(xvar))
    global_n <- nrow(dat_tmp) / length(levels(factor(tmp[[1]])))

    plt1 <- ggplot(out, aes(x = !!sym(xvar), group = 1)) + geom_bar(aes(weight = vol), fill = "gray40") +
        geom_hline(yintercept = global_n, colour = "green", size = line_size) +
        labs(title = paste("Number of claims:", col_names[k]), x = col_names[k], y = "claim counts")

    plt2 <- ggplot(out, aes(x = !!sym(xvar), group = 1)) + geom_bar(aes(weight = avg), fill = "gray60") +
        geom_hline(yintercept = global_avg, colour = "blue", size = line_size) +
        coord_cartesian(ylim = severity_limits) +
        labs(title = paste("Average claim amount:", col_names[k]), x = col_names[k], y = "average claim amount")

    grid.arrange(plt1, plt2, ncol = 2, top = col_names[k])
}

# COMMAND ----------

# DBTITLE 1,Feature correlation
## -----------------------------------------------------------------------------
sel_col <- c("Age","WeeklyPay","HoursWorkedPerWeek","DaysWorkedPerWeek",
             "AccYear","AccMonth","AccWeekday","AccTime","RepDelay")
dat_tmp <- dat[, sel_col]


## -----------------------------------------------------------------------------
corrMat <- round(cor(dat_tmp, method = "pearson"), 2)
corrMat
corrplot(corrMat, method = "color")


## -----------------------------------------------------------------------------
corrMat <- round(cor(dat_tmp, method = "spearman"), 2)
corrMat
corrplot(corrMat, method = "color")

# COMMAND ----------

str(dat)

# COMMAND ----------

#Uncomment if not installed on the cluster
install.packages("mlflow")
library(mlflow)
install_mlflow()

# COMMAND ----------

# DBTITLE 1,Set MlFlow Experiment
mlflow_set_experiment(experiment_id = "363313668622318") #This is for Experiment "GroupDisability"

# COMMAND ----------

# DBTITLE 1,Regression Modeling: General modelling: parameters
## -----------------------------------------------------------------------------
# used/selected features
col_features <- c("AgeNN","GenderNN","DependentChildrenNN","DependentsOtherNN",
                  "WeeklyPayNN","PartTimeFullTimeNN","HoursWorkedPerWeekNN",
                  "DaysWorkedPerWeekNN","AccYearNN","AccMonthNN","AccWeekdayNN",
                  "AccTimeNN","RepDelayNN","Marital1","Marital2","Marital3")
col_names <- c("Age","Gender","DependentChildren","DependentsOther","WeeklyPay",
               "PartTimeFullTime","HoursWorkedPerWeek","DaysWorkedPerWeek",
               "AccYear","AccMonth","AccWeekday","AccTime","RepDelay",
               "Marital1","Marital2","Marital3")


## -----------------------------------------------------------------------------
# select p in [2,3]
p <- 2.5

# COMMAND ----------

library(vctrs)

# COMMAND ----------

#runs <- mlflow_list_run_infos(
#  run_view_type = c("ACTIVE_ONLY"))
#if (vec_size(runs %>% filter(status == "RUNNING")) > 0) {
#  mlflow_end_run()
#}

# COMMAND ----------

# DBTITLE 1,Run Regression Model and log results
with(mlflow_start_run(), {
  ## -----------------------------------------------------------------------------
  # homogeneous model (learn)
  (size_hom <- round(mean(learn$Claim)))
  log_size_hom <- log(size_hom)

  ## -----------------------------------------------------------------------------
  
  mlflow_log_metric("learn_p2", round(gamma_loss(learn$Claim, size_hom), 4))
  mlflow_log_metric("learn_pp", round(p_loss(learn$Claim, size_hom, p) * 10, 4))
  mlflow_log_metric("learn_p3", round(ig_loss(learn$Claim, size_hom) * 1000, 4))
  mlflow_log_metric("test_p2", round(gamma_loss(test$Claim, size_hom), 4))
  mlflow_log_metric("test_pp", round(p_loss(test$Claim, size_hom, p) * 10, 4))
  mlflow_log_metric("test_p3", round(ig_loss(test$Claim, size_hom) * 1000, 4))
  mlflow_log_metric("avg_size", round(size_hom, 0))
  mlflow_set_tag("mlflow.runName", "Null Model")
  #mlflow_log_model(learn, "LR Model")
})

# COMMAND ----------

# DBTITLE 1, Plain-vanilla Neural Network
## -----------------------------------------------------------------------------
# Size of input for neural networks
q0 <- length(col_features)
qqq <- c(q0, c(20,15,10), 1)

sprintf("Neural network with K=3 hidden layer")
sprintf("Input feature dimension: q0 = %s", q0)
sprintf("Number of hidden neurons first layer: q1 = %s", qqq[2])
sprintf("Number of hidden neurons second layer: q2 = %s", qqq[3])
sprintf("Number of hidden neurons third layer: q3 = %s", qqq[4])
sprintf("Output dimension: %s", qqq[5])


## -----------------------------------------------------------------------------
# matrices
YY <- as.matrix(as.numeric(learn$Claim))
XX <- as.matrix(learn[, col_features]) 
TT <- as.matrix(test[, col_features])

# COMMAND ----------

# DBTITLE 1,Plain-vanilla Gamma Neural Network
## -----------------------------------------------------------------------------
Design  <- layer_input(shape = c(qqq[1]), dtype = 'float32', name = 'design')

Output <- Design %>%    
    layer_dense(units = qqq[2], activation = 'tanh', name = 'layer1') %>%
    layer_dense(units = qqq[3], activation = 'tanh', name = 'layer2') %>%
    layer_dense(units = qqq[4], activation = 'tanh', name = 'layer3') %>%
    layer_dense(units = 1, activation = 'exponential', name = 'output', 
                weights = list(array(0, dim = c(qqq[4], 1)), array(log_size_hom, dim = c(1))))

model_p2 <- keras_model(inputs = list(Design), outputs = c(Output))

# COMMAND ----------

# DBTITLE 0,Compilation
## -----------------------------------------------------------------------------
model_p2 %>% compile(
    loss = k_gamma_loss,
    optimizer = 'nadam'
)

summary(model_p2)

# COMMAND ----------

# DBTITLE 0,Fitting
## -----------------------------------------------------------------------------
# set hyperparameters
epochs <- 100
batch_size <- 5000
validation_split <- 0.2 # set to >0 to see train/validation loss in plot(fit)
verbose <- 1


## -----------------------------------------------------------------------------
# store and use only the best model
cp_path <- paste("/dbfs/tmp/Networks/model_p2")

cp_callback <- callback_model_checkpoint(
    filepath = cp_path,
    monitor = "val_loss",
    save_weights_only = TRUE,
    save_best_only = TRUE,
    verbose = 0
)


## -----------------------------------------------------------------------------
fit_p2 <- model_p2 %>%
  fit(
    list(XX), list(YY),
    validation_split = validation_split,
    epochs = epochs,
    batch_size = batch_size,
    callbacks = list(cp_callback),
    verbose = verbose
  )


## -----------------------------------------------------------------------------
plot_result <- plot(fit_p2)
print(typeof(plot_result))
print(plot_result)

## -----------------------------------------------------------------------------
keras_plot_loss_min(fit_p2, seed)


## -----------------------------------------------------------------------------
load_model_weights_hdf5(model_p2, cp_path)

# COMMAND ----------

print(plot_result)

# COMMAND ----------

# DBTITLE 1,Validation
## -----------------------------------------------------------------------------
# calculating the predictions
with(mlflow_start_run(), {
  mlflow_log_param("epochs", epochs)
  mlflow_log_param("batch_size", batch_size)
  mlflow_log_param("validation_split", validation_split)
  learn$fitshp2 <- as.vector(model_p2 %>% predict(list(XX)))
  test$fitshp2 <- as.vector(model_p2 %>% predict(list(TT)))

  # average in-sample and out-of-sample losses (in 10^(0))
  sprintf("Gamma deviance shallow network (train): %s", round(gamma_loss(learn$Claim, learn$fitshp2), 4))
  sprintf("Gamma deviance shallow network (test): %s", round(gamma_loss(test$Claim, test$fitshp2), 4))

  # average claims size
  sprintf("Average size (test): %s", round(mean(test$fitshp2), 1))

  model = "Plain-vanilla p2 (gamma)"
  learn_p2 <- round(gamma_loss(learn$Claim, learn$fitshp2), 4)
   learn_pp <- round(p_loss(learn$Claim, learn$fitshp2, p) * 10, 4)
   learn_p3 <- round(ig_loss(learn$Claim, learn$fitshp2) * 1000, 4)
   test_p2 <- round(gamma_loss(test$Claim, test$fitshp2), 4)
   test_pp <- round(p_loss(test$Claim, test$fitshp2, p) * 10, 4)
   test_p3 <- round(ig_loss(test$Claim, test$fitshp2) * 1000, 4)
   avg_size <-  round(mean(test$fitshp2), 0)
  ## -----------------------------------------------------------------------------
  mlflow_log_metric("learn_p2", learn_p2)
  mlflow_log_metric("learn_pp", learn_pp)
  mlflow_log_metric("learn_p3", learn_p3)
  mlflow_log_metric("test_p2", test_p2)
  mlflow_log_metric("test_pp", test_pp)
  mlflow_log_metric("test_p3", test_p3)
  mlflow_log_metric("avg_size", avg_size)
  mlflow_set_tag("mlflow.runName", "Plain-vanilla p2 (gamma)")
  mlflow_log_model(model_p2, "model") 
})

# COMMAND ----------


## -----------------------------------------------------------------------------
# Age
plt1 <- plot_size(test, "Age", "Claim size by Age", "shp2", "fitshp2")
# Gender
plt2 <- plot_size(test, "Gender", "Claim size by Gender", "shp2", "fitshp2")
# AccMonth
plt3 <- plot_size(test, "AccMonth", "Claim size by AccMonth", "shp2", "fitshp2")
# AccYear
plt4 <- plot_size(test, "AccYear", "Claim size by AccYear", "shp2", "fitshp2")

grid.arrange(plt1, plt2, plt3, plt4)


## -----------------------------------------------------------------------------
Design  <- layer_input(shape = c(qqq[1]), dtype = 'float32', name = 'design')

Output <- Design %>%    
    layer_dense(units=qqq[2], activation='tanh', name='layer1') %>%
    layer_dense(units=qqq[3], activation='tanh', name='layer2') %>%
    layer_dense(units=qqq[4], activation='tanh', name='layer3') %>%
    layer_dense(units=1, activation='exponential', name='output', 
                weights=list(array(0, dim=c(qqq[4],1)), array(log_size_hom, dim=c(1))))

model_pp <- keras_model(inputs = list(Design), outputs = c(Output))


## -----------------------------------------------------------------------------
model_pp %>% compile(
    loss = k_p_loss,
    optimizer = 'nadam'
)

summary(model_pp)

# COMMAND ----------

## -----------------------------------------------------------------------------
# set hyperparameters
epochs <- 100
batch_size <- 5000
validation_split <- 0.2 # set to >0 to see train/validation loss in plot(fit)
verbose <- 1


## -----------------------------------------------------------------------------
# store and use only the best model
cp_path <- paste("/dbfs/tmp/Networks/model_pp")

cp_callback <- callback_model_checkpoint(
    filepath = cp_path,
    monitor = "val_loss",
    save_weights_only = TRUE,
    save_best_only = TRUE,
    verbose = 0
)


## -----------------------------------------------------------------------------
fit_pp <- model_pp %>% fit(
    list(XX), list(YY),
    validation_split = validation_split,
    epochs = epochs,
    batch_size = batch_size,
    callbacks = list(cp_callback),  
    verbose = verbose
)


## -----------------------------------------------------------------------------
plot(fit_pp)


## -----------------------------------------------------------------------------
keras_plot_loss_min(fit_pp, seed)


## -----------------------------------------------------------------------------
load_model_weights_hdf5(model_pp, cp_path)

# COMMAND ----------

# DBTITLE 1,Plain Vanilla Model PP
## -----------------------------------------------------------------------------
# calculating the predictions
with(mlflow_start_run(), {
  mlflow_log_param("epochs", epochs)
  mlflow_log_param("batch_size", batch_size)
  mlflow_log_param("validation_split", validation_split)
  learn$fitshpp <- as.vector(model_pp %>% predict(list(XX)))
  test$fitshpp <- as.vector(model_pp %>% predict(list(TT)))

  # average in-sample and out-of-sample losses (in 10^(0))
  sprintf("p-loss deviance shallow network (train): %s", round(p_loss(learn$Claim, learn$fitshpp, p), 4))
  sprintf("p-loss deviance shallow network (test): %s", round(p_loss(test$Claim, test$fitshpp, p), 4))

  # average claims size
  sprintf("Average size (test): %s", round(mean(test$fitshpp), 1))

 learn_p2 <- round(gamma_loss(learn$Claim, learn$fitshpp), 4)
 learn_pp <- round(p_loss(learn$Claim, learn$fitshpp, p) * 10, 4)
 learn_p3 <- round(ig_loss(learn$Claim, learn$fitshpp) * 1000, 4)
 test_p2 <- round(gamma_loss(test$Claim, test$fitshpp), 4)
 test_pp <- round(p_loss(test$Claim, test$fitshpp, p) * 10, 4)
 test_p3 <- round(ig_loss(test$Claim, test$fitshpp) * 1000, 4)
 avg_size <- round(mean(test$fitshpp), 0)
 model = paste0("Plain-vanilla pp (p=", p,")")
  ## -----------------------------------------------------------------------------
  mlflow_log_metric("learn_p2", learn_p2)
  mlflow_log_metric("learn_pp", learn_pp)
  mlflow_log_metric("learn_p3", learn_p3)
  mlflow_log_metric("test_p2", test_p2)
  mlflow_log_metric("test_pp", test_pp)
  mlflow_log_metric("test_p3", test_p3)
  mlflow_log_metric("avg_size", avg_size)
  mlflow_set_tag("mlflow.runName", model)
  mlflow_log_model(model_pp, "model") 
})

# COMMAND ----------

## -----------------------------------------------------------------------------
# Age
plt1 <- plot_size(test, "Age", "Claim size by Age", "shpp", "fitshpp")
# Gender
plt2 <- plot_size(test, "Gender", "Claim size by Gender", "shpp", "fitshpp")
# AccMonth
plt3 <- plot_size(test, "AccMonth", "Claim size by AccMonth", "shpp", "fitshpp")
# AccYear
plt4 <- plot_size(test, "AccYear", "Claim size by AccYear", "shpp", "fitshpp")

grid.arrange(plt1, plt2, plt3, plt4)

# COMMAND ----------

# DBTITLE 1,Inverse Gaussian Neural Network (p=3)
## -----------------------------------------------------------------------------
Design  <- layer_input(shape = c(qqq[1]), dtype = 'float32', name = 'design')

Output <- Design %>%    
    layer_dense(units=qqq[2], activation='tanh', name='layer1') %>%
    layer_dense(units=qqq[3], activation='tanh', name='layer2') %>%
    layer_dense(units=qqq[4], activation='tanh', name='layer3') %>%
    layer_dense(units=1, activation='exponential', name='output', 
                weights=list(array(0, dim=c(qqq[4],1)), array(log_size_hom, dim=c(1))))

model_p3 <- keras_model(inputs = list(Design), outputs = c(Output))

# COMMAND ----------

# DBTITLE 1,Compilation
## -----------------------------------------------------------------------------
model_p3 %>% compile(
    loss = k_ig_loss,
    optimizer = 'nadam'
)

summary(model_p3)

# COMMAND ----------

# DBTITLE 1,Fitting
## -----------------------------------------------------------------------------
# set hyperparameters
epochs <- 100
batch_size <- 5000
validation_split <- 0.2 # set to >0 to see train/validation loss in plot(fit)
verbose <- 1


## -----------------------------------------------------------------------------
# store and use only the best model
cp_path <- paste("/dbfs/tmp/Networks/model_p3")

cp_callback <- callback_model_checkpoint(
    filepath = cp_path,
    monitor = "val_loss",
    save_weights_only = TRUE,
    save_best_only = TRUE,
    verbose = 0
)


## -----------------------------------------------------------------------------
fit_p3 <- model_p3 %>% fit(
    list(XX), list(YY),
    validation_split = validation_split,
    epochs = epochs,
    batch_size = batch_size,
    callbacks = list(cp_callback),
    verbose = verbose
)

# COMMAND ----------

# DBTITLE 1,Plot
## -----------------------------------------------------------------------------
plot(fit_p3)


## -----------------------------------------------------------------------------
keras_plot_loss_min(fit_p3, seed)

# COMMAND ----------

## -----------------------------------------------------------------------------
load_model_weights_hdf5(model_p3, cp_path)

# COMMAND ----------

# DBTITLE 1,Validation
## -----------------------------------------------------------------------------
# calculating the predictions
with(mlflow_start_run(), {
  mlflow_log_param("epochs", epochs)
  mlflow_log_param("batch_size", batch_size)
  mlflow_log_param("validation_split", validation_split)
  learn$fitshp3 <- as.vector(model_p3 %>% predict(list(XX)))
  test$fitshp3 <- as.vector(model_p3 %>% predict(list(TT)))

  # average in-sample and out-of-sample losses (in 10^(0))
  sprintf("IG deviance shallow network (train): %s", round(ig_loss(learn$Claim, learn$fitshp3), 4))
  sprintf("IG deviance shallow network (test): %s", round(ig_loss(test$Claim, test$fitshp3), 4))

  # average claims size
  sprintf("Average size (test): %s", round(mean(test$fitshp3), 1))
 model = "Plain-vanilla p3 (inverse gaussian)"
 learn_p2 <- round(gamma_loss(learn$Claim, learn$fitshp3), 4)
 learn_pp <- round(p_loss(learn$Claim, learn$fitshp3, p) * 10, 4)
 learn_p3 <- round(ig_loss(learn$Claim, learn$fitshp3) * 1000, 4)
 test_p2 <- round(gamma_loss(test$Claim, test$fitshp3), 4)
 test_pp <- round(p_loss(test$Claim, test$fitshp3, p) * 10, 4)
 test_p3 <- round(ig_loss(test$Claim, test$fitshp3) * 1000, 4)
 avg_size <- round(mean(test$fitshp3), 0)
  ## -----------------------------------------------------------------------------
  mlflow_log_metric("learn_p2", learn_p2)
  mlflow_log_metric("learn_pp", learn_pp)
  mlflow_log_metric("learn_p3", learn_p3)
  mlflow_log_metric("test_p2", test_p2)
  mlflow_log_metric("test_pp", test_pp)
  mlflow_log_metric("test_p3", test_p3)
  mlflow_log_metric("avg_size", avg_size)
  mlflow_set_tag("mlflow.runName", "Plain-vanilla p3 (inverse gaussian)")
  mlflow_log_model(model_p2, "model")
})

# COMMAND ----------

# DBTITLE 1,Calibration
## -----------------------------------------------------------------------------
  # Age
  plt1 <- plot_size(test, "Age", "Claim size by Age", "shp3", "fitshp3")
  # Gender
  plt2 <- plot_size(test, "Gender", "Claim size by Gender", "shp3", "fitshp3")
  # AccMonth
  plt3 <- plot_size(test, "AccMonth", "Claim size by AccMonth", "shp3", "fitshp3")
  # AccYear
  plt4 <- plot_size(test, "AccYear", "Claim size by AccYear", "shp3", "fitshp3")

  grid.arrange(plt1, plt2, plt3, plt4)

# COMMAND ----------

# MAGIC %md
# MAGIC [Link to Experiment](https://e2-demo-field-eng.cloud.databricks.com/?o=1444828305810485#mlflow/experiments/83447045894155/s?searchInput=&orderByKey=metrics.%60Average%20size%20%28test%29%60&orderByAsc=false&startTime=LAST_HOUR&lifecycleFilter=Active&modelVersionFilter=All%20Runs&categorizedUncheckedKeys[attributes][]=&categorizedUncheckedKeys[params][]=&categorizedUncheckedKeys[metrics][]=&categorizedUncheckedKeys[tags][]=&diffSwitchSelected=false&preSwitchCategorizedUncheckedKeys[attributes][]=&preSwitchCategorizedUncheckedKeys[params][]=&preSwitchCategorizedUncheckedKeys[metrics][]=&preSwitchCategorizedUncheckedKeys[tags][]=&postSwitchCategorizedUncheckedKeys[attributes]=,Version&postSwitchCategorizedUncheckedKeys[params][]=&postSwitchCategorizedUncheckedKeys[metrics][]=&postSwitchCategorizedUncheckedKeys[tags][]=)

# COMMAND ----------

# MAGIC %md
# MAGIC We can draw the following conclusions:
# MAGIC 
# MAGIC - Fitting these networks takes in average ~30 epochs, thus, fitting is very fast here.
# MAGIC - We give preference to the gamma model p=2. However, these solutions would need further analysis to perform a thorough model selection, e.g., one can study Tukey{Anscombe plots, dispersion parameters, etc. We refrain from doing so because this is not the main purpose of this notebook.
# MAGIC - We remark that the inverse Gaussian model does not have the lowest in-sample loss on Lp=3. This seems counter-intuitive, but it is caused by the fact that we exercise early stopping. In fact, the inverse Gaussian model is more sensitive in fitting and, typically, this results in an earlier stopping time. Here, it uses less than 30 epochs, whereas the other two cases use more than 30 epochs. In general, the inverse Gaussian model is more difficult to fit.

# COMMAND ----------

# DBTITLE 1,LocalGLMNet: Common neural network specifications
## -----------------------------------------------------------------------------
# Size of input for neural networks
q0 <- length(col_features)
qqq <- c(q0, c(20, 15, 10), 1)

sprintf("Neural network with K=3 hidden layer")
sprintf("Input feature dimension: q0 = %s", q0)
sprintf("Number of hidden neurons first layer: q1 = %s", qqq[2])
sprintf("Number of hidden neurons second layer: q2 = %s", qqq[3])
sprintf("Number of hidden neurons third layer: q3 = %s", qqq[4])
sprintf("Number of hidden neurons third layer: q4 = %s", qqq[1])
sprintf("Output dimension: %s", qqq[5])


## -----------------------------------------------------------------------------
# matrices
YY <- as.matrix(as.numeric(learn$Claim))
XX <- as.matrix(learn[, col_features])
TT <- as.matrix(test[, col_features])


## -----------------------------------------------------------------------------
# neural network structure
Design  <- layer_input(shape = c(qqq[1]), dtype = 'float32', name = 'design') 

Attention <- Design %>%    
    layer_dense(units=qqq[2], activation='tanh', name='layer1') %>%
    layer_dense(units=qqq[3], activation='tanh', name='layer2') %>%
    layer_dense(units=qqq[4], activation='tanh', name='layer3') %>%
    layer_dense(units=qqq[1], activation='linear', name='attention')

Output <- list(Design, Attention) %>% layer_dot(name='LocalGLM', axes=1) %>% 
    layer_dense(
      units=1, activation='exponential', name='output',
      weights=list(array(0, dim=c(1,1)), array(log_size_hom, dim=c(1)))
    )

# COMMAND ----------

#runs <- mlflow_list_run_infos(
#  run_view_type = c("ACTIVE_ONLY"))
#if (vec_size(runs %>% filter(status == "RUNNING")) > 0) {
#  mlflow_end_run()
#}

# COMMAND ----------

## -----------------------------------------------------------------------------
  model_lgn_p2 <- keras_model(inputs = list(Design), outputs = c(Output))


  ## -----------------------------------------------------------------------------
  model_lgn_p2 %>% compile(
      loss = k_gamma_loss,
      optimizer = 'nadam'
  )
  summary(model_lgn_p2)
  ## -----------------------------------------------------------------------------
  # set hyperparameters
  epochs <- 100
  batch_size <- 5000
  validation_split <- 0.2 # set to >0 to see train/validation loss in plot(fit)
  verbose <- 1
  
  ## -----------------------------------------------------------------------------
  # store and use only the best model
  cp_path <- paste("/dbfs/tmp/Networks/model_lgn_p2")

  cp_callback <- callback_model_checkpoint(
      filepath = cp_path,
      monitor = "val_loss",
      save_weights_only = TRUE,
      save_best_only = TRUE,
      verbose = 0
  )


  ## -----------------------------------------------------------------------------
  fit_lgn_p2 <- model_lgn_p2 %>% fit(
      list(XX), list(YY),
      validation_split = validation_split,
      epochs = epochs,
      batch_size = batch_size,
      callbacks = list(cp_callback),
      verbose = verbose
  )

# COMMAND ----------

## -----------------------------------------------------------------------------
  plot(fit_lgn_p2) 
  ## -----------------------------------------------------------------------------

# COMMAND ----------

keras_plot_loss_min(fit_lgn_p2, seed)
#  mlflow_log_artifact("/dbfs/tmp/keras_plot_loss.png")

# COMMAND ----------

## -----------------------------------------------------------------------------
  load_model_weights_hdf5(model_lgn_p2, cp_path)
  ## -----------------------------------------------------------------------------

# COMMAND ----------

# DBTITLE 1,LocalGLMnet Gamma (p=2)
with(mlflow_start_run(), {
  mlflow_log_param("epochs", epochs)
  mlflow_log_param("batch_size", batch_size)
  mlflow_log_param("validation_split", validation_split)
  # calculating the predictions
  learn$fitlgnp2 <- as.vector(model_lgn_p2 %>% predict(list(XX)))
  test$fitlgnp2 <- as.vector(model_lgn_p2 %>% predict(list(TT)))

  # average in-sample and out-of-sample losses (in 10^(0))
  sprintf("Gamma deviance shallow network (train): %s", round(gamma_loss(learn$Claim, learn$fitlgnp2), 4))
  sprintf("Gamma deviance shallow network (test): %s", round(gamma_loss(test$Claim, test$fitlgnp2), 4))

  # average claims size
  sprintf("Average size (test): %s", round(mean(test$fitlgnp2), 1))

   learn_p2 <- round(gamma_loss(learn$Claim, learn$fitlgnp2), 4)
   learn_pp <- round(p_loss(learn$Claim, learn$fitlgnp2, p) * 10, 4)
   learn_p3 <- round(ig_loss(learn$Claim, learn$fitlgnp2) * 1000, 4)
   test_p2 <- round(gamma_loss(test$Claim, test$fitlgnp2), 4)
   test_pp <- round(p_loss(test$Claim, test$fitlgnp2, p) * 10, 4)
   test_p3 <- round(ig_loss(test$Claim, test$fitlgnp2) * 1000, 4)
   avg_size <- round(mean(test$fitlgnp2), 0)
  mlflow_log_metric("learn_p2", learn_p2)
  mlflow_log_metric("learn_pp", learn_pp)
  mlflow_log_metric("learn_p3", learn_p3)
  mlflow_log_metric("test_p2", test_p2)
  mlflow_log_metric("test_pp", test_pp)
  mlflow_log_metric("test_p3", test_p3)
  mlflow_log_metric("avg_size", avg_size)
  mlflow_set_tag("mlflow.runName", "LocalGLMnet p2 (gamma)")
  mlflow_log_model(model_lgn_p2, "model") 
})

# COMMAND ----------

# DBTITLE 1,Plot
## -----------------------------------------------------------------------------
# Age
plt1 <- plot_size(test, "Age", "Claim size by Age", "lgnp2", "fitlgnp2")
# Gender
plt2 <- plot_size(test, "Gender", "Claim size by Gender", "lgnp2", "fitlgnp2")
# AccMonth
plt3 <- plot_size(test, "AccMonth", "Claim size by AccMonth", "lgnp2", "fitlgnp2")
# AccYear
plt4 <- plot_size(test, "AccYear", "Claim size by AccYear", "lgnp2", "fitlgnp2")

grid.arrange(plt1, plt2, plt3, plt4)

# COMMAND ----------

# DBTITLE 1,Compilation
## -----------------------------------------------------------------------------
model_lgn_pp <- keras_model(inputs = list(Design), outputs = c(Output))


## -----------------------------------------------------------------------------
model_lgn_pp %>% compile(
    loss = k_p_loss,
    optimizer = 'nadam'
)
summary(model_lgn_pp)

# COMMAND ----------

# DBTITLE 1,Fitting
## -----------------------------------------------------------------------------
# set hyperparameters
epochs <- 100
batch_size <- 5000
validation_split <- 0.2 # set to >0 to see train/validation loss in plot(fit)
verbose <- 1


## -----------------------------------------------------------------------------
# store and use only the best model
cp_path <- paste("/dbfs/tmp/model_lgn_pp/cp")

cp_callback <- callback_model_checkpoint(
    filepath = cp_path,
    monitor = "val_loss",
    save_weights_only = TRUE,
    save_best_only = TRUE,
    verbose = 0
)

# COMMAND ----------

## -----------------------------------------------------------------------------
fit_lgn_pp <- model_lgn_pp %>% fit(
    list(XX), list(YY),
    validation_split = validation_split,
    epochs = epochs,
    batch_size = batch_size,
    callbacks = list(cp_callback),
    verbose = verbose
)

# COMMAND ----------

## -----------------------------------------------------------------------------
plot(fit_lgn_pp)

# COMMAND ----------

## -----------------------------------------------------------------------------
keras_plot_loss_min(fit_lgn_pp, seed)

# COMMAND ----------

## -----------------------------------------------------------------------------
load_model_weights_hdf5(model_lgn_pp, cp_path)

# COMMAND ----------

# DBTITLE 1,Validation
with(mlflow_start_run(), {
  mlflow_log_param("epochs", epochs)
  mlflow_log_param("batch_size", batch_size)
  mlflow_log_param("validation_split", validation_split)
  ## -----------------------------------------------------------------------------
  # calculating the predictions
  learn$fitlgnpp <- as.vector(model_pp %>% predict(list(XX)))
  test$fitlgnpp <- as.vector(model_pp %>% predict(list(TT)))

  # average in-sample and out-of-sample losses (in 10^(0))
  sprintf("p-loss deviance shallow network (train): %s", round(p_loss(learn$Claim, learn$fitlgnpp, p), 4))
  sprintf("p-loss deviance shallow network (test): %s", round(p_loss(test$Claim, test$fitlgnpp, p), 4))

  # average claims size
  sprintf("Average size (test): %s", round(mean(test$fitlgnpp), 1))
  model <- "LocalGLMnet pp (p=2.5)"
  learn_p2 <- round(gamma_loss(learn$Claim, learn$fitlgnpp), 4)
  learn_pp <- round(p_loss(learn$Claim, learn$fitlgnpp, p) * 10, 4)
  learn_p3 <- round(ig_loss(learn$Claim, learn$fitlgnpp) * 1000, 4)
  test_p2 <- round(gamma_loss(test$Claim, test$fitlgnpp), 4)
  test_pp <- round(p_loss(test$Claim, test$fitlgnpp, p) * 10, 4)
  test_p3 <- round(ig_loss(test$Claim, test$fitlgnpp) * 1000, 4)
  avg_size <- round(mean(test$fitlgnpp), 0)
  ## -----------------------------------------------------------------------------
  mlflow_log_metric("learn_p2", learn_p2)
  mlflow_log_metric("learn_pp", learn_pp)
  mlflow_log_metric("learn_p3", learn_p3)
  mlflow_log_metric("test_p2", test_p2)
  mlflow_log_metric("test_pp", test_pp)
  mlflow_log_metric("test_p3", test_p3)
  mlflow_log_metric("avg_size", avg_size)
  mlflow_set_tag("mlflow.runName", "LocalGLMnet pp (p=2.5)")
  mlflow_log_model(model_lgn_p2, "model")
})

# COMMAND ----------

# DBTITLE 1,Calibration
## -----------------------------------------------------------------------------
# Age
plt1 <- plot_size(test, "Age", "Claim size by Age", "lgnpp", "fitlgnpp")
# Gender
plt2 <- plot_size(test, "Gender", "Claim size by Gender", "lgnpp", "fitlgnpp")
# AccMonth
plt3 <- plot_size(test, "AccMonth", "Claim size by AccMonth", "lgnpp", "fitlgnpp")
# AccYear
plt4 <- plot_size(test, "AccYear", "Claim size by AccYear", "lgnpp", "fitlgnpp")

grid.arrange(plt1, plt2, plt3, plt4)

# COMMAND ----------

# DBTITLE 1,LocalGlmnet 2<p<3
## -----------------------------------------------------------------------------
model_lgn_p3 <- keras_model(inputs = list(Design), outputs = c(Output))

# COMMAND ----------

# DBTITLE 1,Complation
## -----------------------------------------------------------------------------
model_lgn_p3 %>% compile(
    loss = k_ig_loss,
    optimizer = 'nadam'
)
summary(model_lgn_p3)

# COMMAND ----------

# DBTITLE 1,Fitting
## -----------------------------------------------------------------------------
# set hyperparameters
epochs <- 100
batch_size <- 5000
validation_split <- 0.2 # set to >0 to see train/validation loss in plot(fit)
verbose <- 1


## -----------------------------------------------------------------------------
# store and use only the best model
cp_path <- paste("/dbfs/tmp/model_lgn_p3/cp")

cp_callback <- callback_model_checkpoint(
    filepath = cp_path,
    monitor = "val_loss",
    save_weights_only = TRUE,
    save_best_only = TRUE,
    verbose = 0
)


## -----------------------------------------------------------------------------
fit_lgn_p3 <- model_lgn_p3 %>% fit(
    list(XX), list(YY),
    validation_split = validation_split,
    epochs = epochs,
    batch_size = batch_size,
    callbacks = list(cp_callback),
    verbose = verbose
)


## -----------------------------------------------------------------------------
plot(fit_lgn_p3)

# COMMAND ----------

## -----------------------------------------------------------------------------
keras_plot_loss_min(fit_lgn_p3, seed)

# COMMAND ----------

# DBTITLE 1,Validation
with(mlflow_start_run(), {
  mlflow_log_param("epochs", epochs)
  mlflow_log_param("batch_size", batch_size)
  mlflow_log_param("validation_split", validation_split)
  ## -----------------------------------------------------------------------------
  load_model_weights_hdf5(model_lgn_p3, cp_path)


  ## -----------------------------------------------------------------------------
  # calculating the predictions
  learn$fitlgnp3 <- as.vector(model_lgn_p3 %>% predict(list(XX)))
  test$fitlgnp3 <- as.vector(model_lgn_p3 %>% predict(list(TT)))

  # average in-sample and out-of-sample losses (in 10^(0))
  sprintf("IG deviance shallow network (train): %s", round(ig_loss(learn$Claim, learn$fitlgnp3), 4))
  sprintf("IG deviance shallow network (test): %s", round(ig_loss(test$Claim, test$fitlgnp3), 4))

  # average claims size
  sprintf("Average size (test): %s", round(mean(test$fitlgnp3), 1))

  model = "LocalGLMnet p3 (inverse gaussian)"
  learn_p2 <- round(gamma_loss(learn$Claim, learn$fitlgnp3), 4)
  learn_pp <- round(p_loss(learn$Claim, learn$fitlgnp3, p) * 10, 4)
  learn_p3 <- round(ig_loss(learn$Claim, learn$fitlgnp3) * 1000, 4)
  test_p2 <- round(gamma_loss(test$Claim, test$fitlgnp3), 4)
  test_pp <- round(p_loss(test$Claim, test$fitlgnp3, p) * 10, 4)
  test_p3 <- round(ig_loss(test$Claim, test$fitlgnp3) * 1000, 4)
  avg_size <- round(mean(test$fitlgnp3), 0)
  ## -----------------------------------------------------------------------------
  mlflow_log_metric("learn_p2", learn_p2)
  mlflow_log_metric("learn_pp", learn_pp)
  mlflow_log_metric("learn_p3", learn_p3)
  mlflow_log_metric("test_p2", test_p2)
  mlflow_log_metric("test_pp", test_pp)
  mlflow_log_metric("test_p3", test_p3)
  mlflow_log_metric("avg_size", avg_size)
  mlflow_set_tag("mlflow.runName", "LocalGLMnet p3 (inverse gaussian")
  mlflow_log_model(model_lgn_p2, "model") 
})

# COMMAND ----------

## -----------------------------------------------------------------------------
# Age
plt1 <- plot_size(test, "Age", "Claim size by Age", "lgnp3", "fitlgnp3")
# Gender
plt2 <- plot_size(test, "Gender", "Claim size by Gender", "lgnp3", "fitlgnp3")
# AccMonth
plt3 <- plot_size(test, "AccMonth", "Claim size by AccMonth", "lgnp3", "fitlgnp3")
# AccYear
plt4 <- plot_size(test, "AccYear", "Claim size by AccYear", "lgnp3", "fitlgnp3")

grid.arrange(plt1, plt2, plt3, plt4, top="LocalGLMnet Inverse Gaussian")

# COMMAND ----------

# MAGIC %md
# MAGIC [Link to Experiment](https://e2-demo-field-eng.cloud.databricks.com/?o=1444828305810485#mlflow/experiments/83447045894155/s?searchInput=&orderByKey=metrics.%60Average%20size%20%28test%29%60&orderByAsc=false&startTime=LAST_HOUR&lifecycleFilter=Active&modelVersionFilter=All%20Runs&categorizedUncheckedKeys[attributes][]=&categorizedUncheckedKeys[params][]=&categorizedUncheckedKeys[metrics][]=&categorizedUncheckedKeys[tags][]=&diffSwitchSelected=false&preSwitchCategorizedUncheckedKeys[attributes][]=&preSwitchCategorizedUncheckedKeys[params][]=&preSwitchCategorizedUncheckedKeys[metrics][]=&preSwitchCategorizedUncheckedKeys[tags][]=&postSwitchCategorizedUncheckedKeys[attributes]=,Version&postSwitchCategorizedUncheckedKeys[params][]=&postSwitchCategorizedUncheckedKeys[metrics][]=&postSwitchCategorizedUncheckedKeys[tags][]=)

# COMMAND ----------

# DBTITLE 1,Conclusion
# MAGIC %md
# MAGIC We can draw the following conclusions:
# MAGIC - We again prefer the gamma model over the other power variance parameter models. 
# MAGIC - Moreover, for this particular data set the LocalGLMnet outperforms the deep FFN network. However, this is not the crucial point of introducing the LocalGLMnet, but the LocalGLMnet leads to interpretable predictions and it allows for variable selection as we are going to demonstrate in the next sections.
