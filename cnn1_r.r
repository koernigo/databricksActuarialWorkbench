# Databricks notebook source
# MAGIC %md
# MAGIC # Disclaimer
# MAGIC This notebook was created for the SAV block course "Deep Learning with Actuarial Applications in R".
# MAGIC 
# MAGIC The course is based on the publications on the following website: https://www.actuarialdatascience.org/
# MAGIC 
# MAGIC Author: Daniel Meier
# MAGIC 
# MAGIC # Convolutional Neural Networks for detection of distorted mortality rates due to errors and migration in population data
# MAGIC # Abstract
# MAGIC Convolutional Neural Networks (CNNs) are typically applied on image and video data for classification problems. A famous CNN winning the ImageNet Recognition Challenge is for example AlexNet, an 8-layer CNN for image classification. CNNs are also frequently applied in Computer Vision for object detection (existence and location) in images. This notebook shows how a simple 4-layer CNN (not counting the batch normalizations separately) can help to detect distortions of mortality rates (which can be considered as a 2D image with dimensions age and year and color channels for males, females, females less males, analogous to red, green, blue) due to errors and migration in population data.
# MAGIC 
# MAGIC # Introduction
# MAGIC Mortality rates $q_{x,t}$ by country and sex, i.e. the probably of dying at age $x$ (last birthday) between year $t$ and $t+1$, are derived from population numbers/exposures $E_{x,t}$ at given points in time $t$, e.g. from census data every 5 years, and death counts $D_{x,t}$, which typically are available at much better time resolution. Both $E_{x,t}$ and $D_{x,t}$ can be affected by errors and migration, which distorts mortality rates $q_{x,t}=D_{x,t}/E_{x,t}$.
# MAGIC 
# MAGIC A way to measure this potential distortion is by considering normalized residuals $r_{x,t}$ defined as
# MAGIC 
# MAGIC $$r_{x,t} = \frac{E_{x,t}-E_{x-1,t-1}(1-q_{x-1,t-1})}{E_{x,t}}$$
# MAGIC i.e. by comparing the actual population numbers/exposures $E_{x,t}$ to the one derived from the previous year $E_{x-1,t-1}(1-q_{x-1,t-1})$. A value $r_{x,t}>0$ indicates immigration or an error. A value $r_{x,t}<0$ indicates emigration or an error.
# MAGIC 
# MAGIC This notebook applies a Computer Vision approach to all mortality rates available from the Human Mortality Database, where inputs $X$ are moving windows of size nx=10 times nt=10 and stepsize sx=5 and st=5 of (logit of) mortality rates $q_{x,t}$ for both sexes (males, females and the difference between females and males are used as channels, analogous to red/green/blue for color images), i.e. $X\in\mathbb{R}^{\text{#windows}\times 10\times 10\times 3}$ and outputs $Y\in\mathbb{R}^{\text{#windows}}$ are maximum absolute values of $r_{x,t}$ over the same moving windows. Whenever the maxima of a given window exceeds the 95% quantile of maxima over all windows, we define that error/migration might be present in the given window.
# MAGIC 
# MAGIC The trained CNN can then be used to detect areas/windows of $q_{x,t}$, where errors and migration potentially distorted mortality rates.
# MAGIC 
# MAGIC The used CNN is a simple 3-layer network comprising
# MAGIC 
# MAGIC * a convolutional 2D layer: 16 filters of size 3 times 3 and stepsize 1 and 1, 3 channels for logit mortality rates of males, females and females less males,
# MAGIC * a convolutional 2D layer: 32 filters of size 3 times 3 and stepsize 1 and 1, 3 channels,
# MAGIC * a convolutional 2D layer: 64 filters of size 3 times 3 and stepsize 1 and 1, 3 channels,
# MAGIC * a fully connected layer.
# MAGIC 
# MAGIC We first formulate the problem as a regression problem minimizing mean square errors, i.e. we would like to predict the size of errors and migration. Then, in order to assess the quality of resulting classifications we use area under curve (AUC).
# MAGIC 
# MAGIC # 0. Import modules, definition of parameters

# COMMAND ----------


options(encoding = 'UTF-8')

# Loading all the necessary packages
library("repr")  # not needed in the Rmarkdown version, only for Jupyter notebook
library("abind")
library("pROC")
library("grid")
library("fields")
library("ggplot2")
library("plotly")
library("keras")
library("tensorflow")


# COMMAND ----------

# MAGIC %md

# COMMAND ----------


knitr::opts_chunk$set(fig.width = 9, fig.asp = 1)
#options(repr.plot.width=4, repr.plot.height=10)


# COMMAND ----------

# MAGIC %md

# COMMAND ----------


pops <- c('AUS','AUT','BEL','BGR','BLR','CAN','CHE','CHL','CZE',
        'DEU','DNK','ESP','EST','FIN','FRA','GBR','GRC','HKG',
        'HRV','HUN','IRL','ISL','ISR','ITA','JPN','KOR','LTU',
        'LUX','LVA','NLD','NOR','NZL','POL','PRT','RUS','SVK',
        'SVN','SWE','TWN','UKR','USA')
nx <- 10 # window size in terms of ages
nt <- 10 # window size in terms of years
sx <- 5 # step width of windows in terms of ages
st <- 5 # step width of windows in terms of years
minAge <- 21
maxAge <- 80
testRatio <- 0.15
validationRatio <- 0.15
thresholdQ <- 0.95 # defines migration/error in terms of a quantile threshold
filterSize <- 5
numberFilters <- 16
filterSize1 <- 3
numberFilters1 <- 16
filterSize2 <- 3
numberFilters2 <- 32
filterSize3 <- 3
numberFilters3 <- 64
numberEpochs <- 800
rxm <- list()
rxf <- list()
X <- list()
Y <- list()
dataRoot <- "../../data"


# COMMAND ----------

# MAGIC %md
# MAGIC Load and visualize mortality rates of GBR males, ages 0 to 110, years 1922 to 2016.

# COMMAND ----------


qxm <- as.matrix(read.csv(file.path(dataRoot, "cnn1", "GBR_M.txt"), skip = 1, sep = "", header = TRUE))
knitr::kable(head(qxm))

fig <- plot_ly(z = matrix(as.numeric(qxm[, 4]), nrow = 111)[1:110, ]) %>%
        layout(title = 'Mortality rates GBR males', scene = list(
          xaxis = list(title = 'Year'),
          yaxis = list(title = 'Age'),
          zaxis = list(title = 'qx')
        )) %>%
        add_surface()
fig


# COMMAND ----------

# MAGIC %md
# MAGIC Load and visualize exposures E, i.e. population numbers by age and year of GBR males.

# COMMAND ----------


E <- as.matrix(read.csv(file.path(dataRoot, "cnn1", "GBR.txt"), skip = 1, sep = "", header = TRUE))
knitr::kable(head(E))

fig <- plot_ly(z = matrix(as.numeric(E[, 4]), nrow = 111)[1:110, ]) %>%
        layout(title = 'Exposures GBR males', scene = list(
          xaxis = list(title = 'Year'),
          yaxis = list(title = 'Age'),
          zaxis = list(title = 'E')
        )) %>%
        add_surface()
fig


# COMMAND ----------

# MAGIC %md
# MAGIC The preparation of the model inputs X, the set of 10x10x3 "images", as well as the preparation of the model outputs Y, the set of residuals, can be looked up in detail in the online tutorial at https://github.com/JSchelldorfer/ActuarialDataScience/blob/master/9%20-%20Convolutional%20neural%20network%20case%20studies/cnn1.ipynb. For this course, we skip this step and directly load the results of these preparations.
# MAGIC 
# MAGIC The plots show the outputs Y rearranged into age (x-axis) times years buckets (y-axis).

# COMMAND ----------


for (jPop in 1:length(pops)) {
    pop = pops[jPop]
    
    lqxm = as.matrix(read.csv(paste0(dataRoot, "/cnn1/logit_qx_", pops[jPop], "_m.csv"), sep = ",", header = FALSE))
    lqxf = as.matrix(read.csv(paste0(dataRoot, "/cnn1/logit_qx_", pops[jPop], "_f.csv"), sep = ",", header = FALSE))
       
    rxm[[pop]] = as.matrix(read.csv(paste0(dataRoot, "/cnn1/residuals_", pops[jPop], "_m.csv"), sep = ",", header = FALSE))
    rxf[[pop]] = as.matrix(read.csv(paste0(dataRoot, "/cnn1/residuals_", pops[jPop], "_f.csv"), sep = ",", header = FALSE))
    
    if (is.element(pop, c('JPN','RUS','USA'))) {
        image.plot(t(rxm[[pop]]))
        mtext(line = 2, side = 1, paste(pop, 'males'))
    }
    
    mx <- floor(floor((maxAge - minAge + 1 - nx) / sx + 1))
    mt <- floor(floor((nrow(rxm[[pop]]) - nt) / st + 1))
    
    X[[pop]] <- array(0, dim = c(mx * mt, nx, nt, 3))
    Y[[pop]] <- array(0, dim = c(mx * mt))
    
    for (j in 0:(mx-1)) {
        for (k in 0:(mt-1)) {                                
            # set up logit qx windows of size nt x nx for each population as input X
            # (population x year buckets x age buckets x sex)
            # logit qx of males as first channel:
            X[[pop]][k*mx+j+1, , , 1] <- lqxm[(k*st+1):(k*st+nt), (j*sx+1):(j*sx+nx)]
            # logit qx of females as second channel:
            X[[pop]][k*mx+j+1, , , 2] <- lqxf[(k*st+1):(k*st+nt), (j*sx+1):(j*sx+nx)]
            # logit qx of females less males as third channel:
            X[[pop]][k*mx+j+1, , , 3] <- X[[pop]][(k*mx+j+1), , , 2] - X[[pop]][k*mx+j+1, , , 1]
            # define output Y as the maximum absolute value of normalized residuals over each window of size nt x nx
            Y[[pop]][k*mx+j+1] <- max(0.5 * abs(
              rxm[[pop]][(k*st+1):(k*st+nt), (j*sx+1):(j*sx+nx)] + rxf[[pop]][(k*st+1):(k*st+nt), (j*sx+1):(j*sx+nx)]
            ))
        }
    }
}


# COMMAND ----------

# MAGIC %md

# COMMAND ----------


# normalize X, Y
for (pop in pops) {
    minX1 <- min(X[[pop]][,,,1])
    maxX1 <- max(X[[pop]][,,,1])
    minX2 <- min(X[[pop]][,,,2])
    maxX2 <- max(X[[pop]][,,,2])
    minX3 <- min(X[[pop]][,,,3])
    maxX3 <- max(X[[pop]][,,,3])
    minY <- min(Y[[pop]])
    maxY <- max(Y[[pop]])    
    X[[pop]][,,,1] <- (X[[pop]][,,,1] - minX1) / (maxX1 - minX1)
    X[[pop]][,,,2] <- (X[[pop]][,,,2] - minX2) / (maxX2 - minX2)
    X[[pop]][,,,3] <- (X[[pop]][,,,3] - minX3) / (maxX3 - minX3)
    Y[[pop]] <- (Y[[pop]] - minY) / (maxY - minY)
}
grid.newpage()
grid.raster(X[['GBR']][2,,,], interpolate = FALSE)  # plot as RBG image


# COMMAND ----------

# MAGIC %md
# MAGIC # 1. Setup and train CNN on a selected subset of all populations
# MAGIC 
# MAGIC The full set of countries is quite heterogenous in terms of immigration/error structures. Observe for example the horizontal, diagonal, and vertical structures of residuals for Japan, Russia, USA above. In the following, we work on the largest cluster of countries with similar immigration/error structure and skip how this cluster was derived. For more details see https://github.com/JSchelldorfer/ActuarialDataScience/blob/master/9%20-%20Convolutional%20neural%20network%20case%20studies/cnn1.ipynb
# MAGIC 
# MAGIC **Exercise:** Use other selections of countries and compare AUCs. (Keep an eye on the number of input samples to be sufficiently large.)
# MAGIC 
# MAGIC **Exercise:** Experiment with other structures/parameters of the CNN, e.g. change the number of layers, strides parameters, activation functions, etc. Make use of summary(cnn) to check the dimensions of inputs/outputs of each layer. How are the dimensions affected by strides, padding, kernel_size, number of filters?

# COMMAND ----------


selectedPop <- c('AUS', 'BGR', 'BLR', 'CAN', 'CZE', 'ESP', 'EST', 'FIN',
          'GBR', 'GRC', 'HKG', 'ISL', 'ITA', 'JPN', 'LTU', 'NZL',
          'POL', 'PRT', 'RUS', 'SVK', 'TWN', 'UKR')

allX <- array(numeric(), c(0,10,10,3))
allY <- array(numeric(), c(0))
for (kPop in selectedPop) {
    allX <- abind(allX, X[[kPop]], along = 1)
    allY <- abind(allY, Y[[kPop]], along = 1)
}

set.seed(0)
tf$random$set_seed(0)

testIdx <- runif(length(allY)) < testRatio
testX <- allX[testIdx,,,]
testY <- allY[testIdx]
trainX <- allX[!testIdx,,,]
trainY <- allY[!testIdx]

cnn <- keras_model_sequential() %>% 
  layer_batch_normalization() %>%
  layer_conv_2d(filters = numberFilters, kernel_size = c(filterSize1, filterSize1),
              strides = c(1,1), padding = 'valid', data_format = 'channels_last') %>% 
  layer_batch_normalization() %>%
  layer_activation('relu') %>%
  layer_conv_2d(filters = numberFilters, kernel_size = c(filterSize2, filterSize2),
                strides = c(1,1), padding = 'valid', data_format = 'channels_last') %>% 
  layer_batch_normalization() %>%
  layer_activation('relu') %>%
  layer_conv_2d(filters = numberFilters, kernel_size = c(filterSize3, filterSize3),
                strides = c(1,1), padding = 'valid', data_format = 'channels_last') %>% 
  layer_batch_normalization() %>%
  layer_activation('relu') %>%
  layer_flatten() %>%
  layer_dense(1) %>%
  layer_activation('sigmoid') %>%
  compile(loss = 'mean_squared_error', optimizer = 'sgd')

summary <- cnn %>% fit(
  x = trainX,
  y = trainY,
  epochs = numberEpochs / 4,
  validation_split = validationRatio,
  sample_weight = (0.2 + trainY) / 1.2,
  batch_size = 64,
  verbose = 0
)

plot(summary)

migErr <- testY >= quantile(testY, thresholdQ)
testPred <- predict(cnn, testX)

plot(testPred, testY - testPred[, 1], col = migErr + 5, main = 'Test set of combined populations',
     xlab = 'Prediction P', ylab = 'Residuals Y-P')
plot(testPred, testY, col = migErr + 5, main = 'Test set of combined populations',
     xlab = 'Prediction P', ylab = 'Output Y')

rocobj <- plot.roc(1 * migErr, testPred[, 1], main = "ROC, AUC", ci = TRUE, print.auc = TRUE)
ciobj <- ci.se(rocobj, specificities = seq(0, 1, 0.01))
plot(ciobj, type = "shape")
plot(ci(rocobj, of = "thresholds", thresholds = "best"))
summary(cnn)


# COMMAND ----------

# MAGIC %md
# MAGIC Comparing predictions P and outputs Y.

# COMMAND ----------


allPred <- predict(cnn, allX)
df <- setNames(data.frame(
        rep(1:(length(allY)/11), each = 11),
        rep(1:11, length(allY)/11),
        allY,
        allPred[, 1],
        allY - allPred[, 1]
      ), c('x','y','z1','z2','z3'))

ggplot(df, aes(y, x, fill = z1)) + geom_tile() +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", midpoint = 0) +
  ggtitle('Output Y') + xlab('Age buckets') + ylab('Years/countries')

ggplot(df, aes(y, x, fill = z2)) + geom_tile() +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", midpoint = 0) +
  ggtitle('Prediction P') + xlab('Age buckets') + ylab('Years/countries')

ggplot(df, aes(y, x, fill = z3)) + geom_tile() +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", midpoint = 0) +
  ggtitle('Residuals Y-P') + xlab('Age buckets') + ylab('Years/countries')


# COMMAND ----------

# MAGIC %md
