library(doParallel)
library(caret)
library(Hmisc)
library(tseries)



# Data Setup and Processing
set.seed(123456)

Full_ts <- read.csv("SPX.csv", header = TRUE)
Train_split <- 0.8
L_ts <- 126
L_pred <- 21

Training_ts <- Full_ts[1:(nrow(Full_ts) * Train_split), ]
Training_df <- matrix(NA, nrow = nrow(Training_ts) - L_ts - L_pred, ncol = L_ts)
Training_Y <- numeric(nrow(Training_df))

for (i in 1:nrow(Training_df)) {
  Training_df[i, ] <- Training_ts[i:(i + L_ts -1), 2]
  Training_Y[i] <- sd(Training_ts[(i + L_ts):(i + L_ts + L_pred - 1), 2]) * sqrt(252)
}

Training_df <- cbind(Training_df, Training_Y)
Training_df <- data.frame(Training_df[sample(nrow(Training_df)), ])


Testing_ts <- Full_ts[((nrow(Full_ts) * Train_split) + L_ts): nrow(Full_ts), ]
Testing_df <- matrix(NA, nrow = nrow(Testing_ts) - L_ts - L_pred, ncol = L_ts)
Testing_Y <- numeric(nrow(Testing_df))

for (i in 1:nrow(Testing_df)) {
  Testing_df[i, ] <- Testing_ts[i:(i + L_ts -1), 2]
  Testing_Y[i] <- sd(Testing_ts[(i + L_ts):(i + L_ts + L_pred - 1), 2]) * sqrt(252)
}

Testing_df <- data.frame(cbind(Testing_df, Testing_Y))



# Random Forest

control <- trainControl(method = 'cv',
                        number = 10,
                        search = 'grid',
                        allowParallel = TRUE)

tunegrid <- expand.grid(.mtry = seq(from = 10, to = 120, by = 10))

cl <- makePSOCKcluster(20)
registerDoParallel(cl)

Model_rf <- train(Training_Y ~.,
                  data = Training_df,
                  method = "rf",
                  ntree = 500,
                  tuneGrid = tunegrid)

stopCluster(cl)



# Gradient Boosting

control <- trainControl(method = 'cv',
                        number = 10,
                        search = 'grid',
                        allowParallel = TRUE)

tunegrid <- expand.grid(nrounds = seq(from = 50, to = 300, by = 50), max_depth = 1,
                        eta = seq(from = 0.05, to = 0.30, by = 0.05), gamma = 0,
                        colsample_bytree = 1, min_child_weight = 1, subsample = 1)

Model_gb <- train(as.matrix(Training_df[ , 1:L_ts]),
                  Training_df$Training_Y,
                  method = "xgbTree",
                  tuneGrid = tunegrid,
                  trControl = control)



# Testing

Testing_df2 <- Testing_df[, (L_ts - 20):L_ts]
Simple <- apply(Testing_df2, 1, sd) * sqrt(252)
cor(Simple, Testing_Y)**2
sqrt(mean(((Simple - Testing_Y)**2)))
mean(abs(Simple - Testing_Y))

Decay <- 0.92
Wgts <- numeric(L_ts)
Wgts[1] <- 1
for (i in 2:L_ts) {Wgts[i] <- Wgts[i - 1] * Decay}
Wgts <- rev(Wgts)
Expon <- apply(Testing_df[ , (1:L_ts)], 1, function(x) sqrt(wtd.var(x, Wgts))) * sqrt(252)
cor(Expon, Testing_Y)**2
sqrt(mean(((Expon - Testing_Y)**2)))
mean(abs(Expon - Testing_Y))

GPred <- numeric(nrow(Testing_df))
for (i in 1:nrow(Testing_df)) {
  s_len <- nrow(Training_ts) + 2*L_ts + i - 1
  fit_garch <- garch(Full_ts$Returns[1:(s_len)], order = c(1, 1))
  forc_garch <- predict(fit_garch)
  GPred[i] <- sqrt(forc_garch[nrow(forc_garch), 1])
}
cor(GPred, Testing_Y)**2
sqrt(mean(((GPred - Testing_Y)**2)))
mean(abs(GPred - Testing_Y))

Pred_Y_RF <- predict(Model_rf, newdata = as.matrix(Testing_df[ , 1:L_ts]))
cor(Pred_Y_RF, Testing_Y)**2
sqrt(mean(((Pred_Y_RF - Testing_Y)**2)))
mean(abs(Pred_Y_RF - Testing_Y))

Pred_Y_GB <- predict(Model_gb, newdata = as.matrix(Testing_df[ , 1:L_ts]))
cor(Pred_Y_GB, Testing_Y)**2
sqrt(mean(((Pred_Y_GB - Testing_Y)**2)))
mean(abs(Pred_Y_GB - Testing_Y))

Pred_Y_ENS <- rowMeans(cbind(Pred_Y_RF, Pred_Y_GB))
cor(Pred_Y_ENS, Testing_Y)**2
sqrt(mean(((Pred_Y_ENS - Testing_Y)**2)))
mean(abs(Pred_Y_ENS - Testing_Y))
