# Databricks notebook source
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
