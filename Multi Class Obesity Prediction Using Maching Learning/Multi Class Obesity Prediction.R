# Installation
if (!require("dplyr"))        install.packages("dplyr",        dependencies = TRUE)
if (!require("caret"))        install.packages("caret",        dependencies = TRUE)
if (!require("rpart"))        install.packages("randomForest", dependencies = TRUE)
if (!require("randomForest")) install.packages("randomForest", dependencies = TRUE)
if (!require("C50"))          install.packages("C50",          dependencies = TRUE)
if (!require("e1071"))        install.packages("e1071",        dependencies = TRUE)
if (!require("nnet"))         install.packages("nnet",         dependencies = TRUE)
if (!require("pROC"))         install.packages("pROC",         dependencies = TRUE)
if (!require("xgboost"))      install.packages("xgboost",      dependencies = TRUE)

# Libraries
library(C50)
library(nnet)
library(e1071)
library(dplyr)
library(caret)
library(rpart)
library(pROC)
library(xgboost)
library(randomForest)

# ---------------------------------------------------------------------------------------------------------- #
# NOTICE: Since the test file from Kaggle does not have labels, we will split a proportion of the train.csv  #
#         file and set that portion as testing data-frame.                                                   #
# ---------------------------------------------------------------------------------------------------------- #
# Constants - file paths
WORKING_FOLDER           <- 'D:/MH6211'
DATA_FOLDER              <- 'Dataset'
RESULT_FOLDER            <- 'Results'
DATA_FILE                <- file.path(DATA_FOLDER, 'train.csv')

# Constants - column names
TARGET_COL               <- 'NObeyesdad'
CATEGORICAL_FEATURE_COLS <- c('Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 
                              'SMOKE', 'SCC', 'CALC', 'MTRANS')
NUMERIC_FEATURE_COLS     <- c('Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 
                              'TUE', 'BMI')

# Constants - obesity levels 
OBESITY_LEVELS_ORDERED   <- c("Insufficient_Weight", "Normal_Weight", 
                              "Overweight_Level_I", "Overweight_Level_II", 
                              "Obesity_Type_I", "Obesity_Type_II", "Obesity_Type_III")

# Set working directory to source file directory
setwd(WORKING_FOLDER)

# Functions - Train XGB model
calculate_overall_auc <- function(targets, predictions) {
  targets <- as.factor(targets)
  predictions <- as.factor(predictions)
  unique_classes <- levels(targets)
  auc_values <- c()
  
  # Calculate AUC for each class in one-vs-rest
  for (class in unique_classes) {
    # Convert to binary: 1 for current class, 0 for others
    binary_targets <- ifelse(targets == class, 1, 0)
    binary_predictions <- ifelse(predictions == class, 1, 0)
    
    # Compute ROC curve and AUC
    roc_obj <- roc(binary_targets, binary_predictions, quiet=TRUE)
    auc_values <- c(auc_values, auc(roc_obj))
  }
  mean_auc <- mean(auc_values)
  
  return (mean_auc)
}

train_default_xgb_model <- function(train_df, target_column, model_name) {
  # Get number of classes
  unique_classes <- levels(train_df[[target_column]])
  num_classes <- length(unique_classes)
  
  # Preprocess train data
  scaler   <- preProcess(x=train_df, method = c("center", "scale"))
  train_df <- predict(scaler, train_df) 

  # Label encoding
  for (col in CATEGORICAL_FEATURE_COLS) {
    train_df[[col]] <- as.integer(train_df[[col]])
  }
  
  # Split the data-set
  train_idx <- createDataPartition(y = train_df[[target_column]], p = 0.7, list = FALSE)
  val_df    <- train_df[-train_idx,]
  train_df  <- train_df[train_idx,]
  
  # Convert data to matrix
  train_matrix <- as.matrix(train_df[, !names(train_df) %in% target_column])
  train_label  <- as.integer(train_df[[target_column]]) - 1
  val_matrix   <- as.matrix(val_df[, !names(val_df) %in% target_column])
  val_label    <- as.integer(val_df[[target_column]]) - 1
  dtrain <- xgb.DMatrix(data = train_matrix, label = train_label)
  dval   <- xgb.DMatrix(data = val_matrix, label = val_label)
  
  # Define parameters
  paramsxgb = list(
    booster="gbtree", #default, else gblinear
    eta=0.1, # learning rate (as in GBM) 0.01-0.2
    max_depth= 7, # max depth of a tree: 3-10
    max.leaves = 500,
    gamma=1, # minimum loss reduction  required for a split, default = 0
    subsample=0.8, # denotes the fraction of observations to be random samples for each tree: 
    colsample_bytree=0.3, # denotes the fraction of columns to be random samples for each tree: 
    objective="multi:softmax",
    num_class=num_classes
  )
  
  # Train
  xgb.fit <- xgb.train(
    params=paramsxgb,
    data=dtrain,
    nrounds=10,
    early_stopping_rounds=10,
    watchlist=list(val1=dtrain,val2=dval),
    verbose=1
  )
  
  # Test
  predictions <- predict(xgb.fit, dval)
  
  # Confusion Matrix
  conf_matrix <- confusionMatrix(as.factor(predictions + 1), as.factor(val_label + 1))
  
  # Extract confusion matrix statistics
  conf_stats <- conf_matrix$byClass
  
  # Calculate Precision, Recall, F1-Score for each class and then average them
  accuracy <- sum(predictions == val_label) / length(val_label)   # Accuracy
  precision <- mean(conf_stats[,"Pos Pred Value"], na.rm = TRUE)  # Precision
  recall <- mean(conf_stats[,"Sensitivity"], na.rm = TRUE)        # Recall
  f1 <- mean(conf_stats[,"F1"], na.rm = TRUE)                     # F1 Score
  auc <- calculate_overall_auc(as.factor(val_label + 1), as.factor(predictions + 1))
  
  return (list(
    name = model_name,
    accuracy = accuracy,
    precision = precision,
    recall = recall,
    f1 = f1,
    auc = auc
  ))
}

# Functions - Data partitioning & Pre-processing
preprocess_df <- function(df, feat_eng=TRUE, scale_numerics = FALSE) {
  # Reordering the levels of obesity
  df$NObeyesdad <- factor(df$NObeyesdad, levels=OBESITY_LEVELS_ORDERED)
  
  # Feature engineering
  df$BMI <- df$Weight / round((df$Height * df$Height), 2)
  
  # Rounding numerical features
  for (col in NUMERIC_FEATURE_COLS) {
    df[[col]] <- round(df[[col]], 2)
    if (scale_numerics) { df[[col]] <- c(scale(df[[col]])) }
  }
  
  # Label encode categorical features
  for (col in CATEGORICAL_FEATURE_COLS) {
    df[[col]] <- as.factor(df[[col]])
  }
  
  return (df)
}

random_split_to_train_test <- function(df, ratio=0.75, scale_numerics = FALSE) {
  # Preprocess first
  df.bmi    <- preprocess_df(df, feat_eng=TRUE, scale_numerics = scale_numerics)
  df.bmi    <- df.bmi[, !(names(df.bmi) %in% c('id'))] # Remove ID column
  df.no_bmi <- df.bmi[, !(names(df.bmi) %in% c('BMI'))]
  
  # Then stratified sampling
  train_idx         <- createDataPartition(df.bmi[[TARGET_COL]], p=ratio, list=FALSE)
  df.bmi.train      <- df.bmi[train_idx,]
  df.bmi.test       <- df.bmi[-train_idx,]
  df.no_bmi.train   <- df.no_bmi[train_idx,]
  df.no_bmi.test    <- df.no_bmi[-train_idx,]
  
  # Then Up-sample + Down-sample
  df.bmi.train_up      <- upSample(x = df.bmi.train %>% select(-NObeyesdad),
                          y = as.factor(df.bmi.train[[TARGET_COL]]),
                          yname = TARGET_COL)
  df.bmi.train_down    <- downSample(x = df.bmi.train %>% select(-NObeyesdad),
                          y = as.factor(df.bmi.train[[TARGET_COL]]),
                          yname = TARGET_COL)
  df.no_bmi.train_up   <- upSample(x = df.no_bmi.train %>% select(-NObeyesdad),
                          y = as.factor(df.no_bmi.train[[TARGET_COL]]),
                          yname = TARGET_COL)
  df.no_bmi.train_down <- downSample(x = df.no_bmi.train %>% select(-NObeyesdad),
                          y = as.factor(df.no_bmi.train[[TARGET_COL]]),
                          yname = TARGET_COL)
  
  return (list(
    df.bmi.train         = df.bmi.train,
    df.bmi.train_up      = df.bmi.train_up,
    df.bmi.train_down    = df.bmi.train_down,
    df.bmi.test          = df.bmi.test,
    df.no_bmi.train      = df.no_bmi.train,
    df.no_bmi.train_up   = df.no_bmi.train_up,
    df.no_bmi.train_down = df.no_bmi.train_down,
    df.no_bmi.test       = df.no_bmi.test
  ))
}

# Functions - Evaluation
evaluation_metrics <- function(model, df) {
  # Make predictions
  targets          <- df[[TARGET_COL]]
  predictions      <- predict(model, df, type='class')

  # Calculate accuracy
  accuracy <- sum(predictions == targets) / length(targets)
  
  # Calculate AUC
  auc <- calculate_overall_auc(targets, predictions)
  
  # Calculate by-class precision, recall, f1 scores
  cm          <- caret::confusionMatrix(predictions, targets, mode="everything")
  sensitivity <- cm$byClass[, "Sensitivity"]
  precision   <- cm$byClass[, "Pos Pred Value"]
  f1_score    <- 2 * (precision * sensitivity) / (precision + sensitivity)
  
  # Calculate weighted precision, recall, f1 scores
  class_support <- table(targets)
  weighted_sensitivity <- sum(sensitivity * class_support) / sum(class_support)
  weighted_precision   <- sum(precision * class_support) / sum(class_support)
  weighted_f1          <- sum(f1_score * class_support) / sum(class_support)
  
  return (list(
    auc = auc,
    accuracy = accuracy,
    weighted_sensitivity = weighted_sensitivity,
    weighted_precision = weighted_precision,
    weighted_f1 = weighted_f1
  ))
}

create_evaluation_table <- function(models_list, df) {
  evaluation_results <- lapply(models_list, function(model) {
    evaluation_metrics(model, df)
  })
  
  ## Combine results into a data-frame
  metrics <- do.call(rbind, lapply(evaluation_results, function(x) unlist(x)))
  evaluation_table <- data.frame(
    AUC = metrics[, "auc"],
    Accuracy = metrics[, "accuracy"],
    Sensitivity = metrics[, "weighted_sensitivity"],
    Precision = metrics[, "weighted_precision"],
    F1 = metrics[, "weighted_f1"]
  )
  
  return (evaluation_table)
}

# 1. Load data
df    <- read.csv(DATA_FILE)
split <- random_split_to_train_test(df, scale_numerics = FALSE)

# 2. Gauge baseline models performance
## 2.1. Tree-based models
### Define models list - with BMI feature
bmi.models       <- list(
  "Decision Tree"          = rpart(NObeyesdad~.,        data=split$df.bmi.train),
  "C5 Decision Tree"       = C5.0(NObeyesdad~.,         data=split$df.bmi.train),
  "RandomForest"           = randomForest(NObeyesdad~., data=split$df.bmi.train)
)

bmi.models_ups   <- list(
  "Decision Tree"          = rpart(NObeyesdad~.,        data=split$df.bmi.train_up),
  "C5 Decision Tree"       = C5.0(NObeyesdad~.,         data=split$df.bmi.train_up),
  "RandomForest"           = randomForest(NObeyesdad~., data=split$df.bmi.train_up)
)

bmi.models_downs <- list(
  "Decision Tree"          = rpart(NObeyesdad~.,        data=split$df.bmi.train_down),
  "C5 Decision Tree"       = C5.0(NObeyesdad~.,         data=split$df.bmi.train_down),
  "RandomForest"           = randomForest(NObeyesdad~., data=split$df.bmi.train_down)
)

### Store results
bmi.evaluation_table       <- create_evaluation_table(bmi.models,          split$df.bmi.test)
bmi.evaluation_table_ups   <- create_evaluation_table(bmi.models_ups,      split$df.bmi.test)
bmi.evaluation_table_downs <- create_evaluation_table(bmi.models_downs,    split$df.bmi.test)
write.csv(bmi.evaluation_table,          file.path(RESULT_FOLDER, 'trees.with_feat_eng.csv'))
write.csv(bmi.evaluation_table_ups,      file.path(RESULT_FOLDER, 'trees_upsampled.with_feat_eng.csv'))
write.csv(bmi.evaluation_table_downs,    file.path(RESULT_FOLDER, 'trees_downsampled.with_feat_eng.csv'))

### Define models list - without BMI feature
no_bmi.models       <- list(
  "Decision Tree"          = rpart(NObeyesdad~.,        data=split$df.no_bmi.train),
  "C5 Decision Tree"       = C5.0(NObeyesdad~.,         data=split$df.no_bmi.train),
  "RandomForest"           = randomForest(NObeyesdad~., data=split$df.no_bmi.train)
)

no_bmi.models_ups   <- list(
  "Decision Tree"          = rpart(NObeyesdad~.,        data=split$df.no_bmi.train_up),
  "C5 Decision Tree"       = C5.0(NObeyesdad~.,         data=split$df.no_bmi.train_up),
  "RandomForest"           = randomForest(NObeyesdad~., data=split$df.no_bmi.train_up)
)

no_bmi.models_downs <- list(
  "Decision Tree"          = rpart(NObeyesdad~.,        data=split$df.no_bmi.train_down),
  "C5 Decision Tree"       = C5.0(NObeyesdad~.,         data=split$df.no_bmi.train_down),
  "RandomForest"           = randomForest(NObeyesdad~., data=split$df.no_bmi.train_down)
)

### Store results
no_bmi.evaluation_table       <- create_evaluation_table(no_bmi.models,       split$df.no_bmi.test)
no_bmi.evaluation_table_ups   <- create_evaluation_table(no_bmi.models_ups,   split$df.no_bmi.test)
no_bmi.evaluation_table_downs <- create_evaluation_table(no_bmi.models_downs, split$df.no_bmi.test)
write.csv(no_bmi.evaluation_table,       file.path(RESULT_FOLDER, 'trees.without_feat_eng.csv'))
write.csv(no_bmi.evaluation_table_ups,   file.path(RESULT_FOLDER, 'trees_upsampled.without_feat_eng.csv'))
write.csv(no_bmi.evaluation_table_downs, file.path(RESULT_FOLDER, 'trees_downsampled.without_feat_eng.csv'))

## 2.2. SVM + Logistic Regression
split_scaled <- random_split_to_train_test(df, scale_numerics = TRUE)

### Define models list - with BMI feature
bmi.models       <- list(
  "Logistic Regression"    = multinom(NObeyesdad~.,     data=split_scaled$df.bmi.train),
  "Support Vector Machine" = svm(NObeyesdad~.,          data=split_scaled$df.bmi.train)
)

bmi.models_ups   <- list(
  "Logistic Regression"    = multinom(NObeyesdad~.,     data=split_scaled$df.bmi.train_up),
  "Support Vector Machine" = svm(NObeyesdad~.,          data=split_scaled$df.bmi.train_up)
)

bmi.models_downs <- list(
  "Logistic Regression"    = multinom(NObeyesdad~.,     data=split_scaled$df.bmi.train_down),
  "Support Vector Machine" = svm(NObeyesdad~.,          data=split_scaled$df.bmi.train_down)
)

### Store results
bmi.evaluation_table       <- create_evaluation_table(bmi.models,          split_scaled$df.bmi.test)
bmi.evaluation_table_ups   <- create_evaluation_table(bmi.models_ups,      split_scaled$df.bmi.test)
bmi.evaluation_table_downs <- create_evaluation_table(bmi.models_downs,    split_scaled$df.bmi.test)
write.csv(bmi.evaluation_table,          file.path(RESULT_FOLDER, 'lr_svm.with_feat_eng.csv'))
write.csv(bmi.evaluation_table_ups,      file.path(RESULT_FOLDER, 'lr_svm_upsampled.with_feat_eng.csv'))
write.csv(bmi.evaluation_table_downs,    file.path(RESULT_FOLDER, 'lr_svm_downsampled.with_feat_eng.csv'))

### Define models list - without BMI feature
no_bmi.models       <- list(
  "Logistic Regression"    = multinom(NObeyesdad~.,     data=split_scaled$df.no_bmi.train),
  "Support Vector Machine" = svm(NObeyesdad~.,          data=split_scaled$df.no_bmi.train)
)

no_bmi.models_ups   <- list(
  "Logistic Regression"    = multinom(NObeyesdad~.,     data=split_scaled$df.no_bmi.train_up),
  "Support Vector Machine" = svm(NObeyesdad~.,          data=split_scaled$df.no_bmi.train_up)
)

no_bmi.models_downs <- list(
  "Logistic Regression"    = multinom(NObeyesdad~.,     data=split_scaled$df.no_bmi.train_down),
  "Support Vector Machine" = svm(NObeyesdad~.,          data=split_scaled$df.no_bmi.train_down)
)

### Store results
no_bmi.evaluation_table       <- create_evaluation_table(no_bmi.models,          split_scaled$df.no_bmi.test)
no_bmi.evaluation_table_ups   <- create_evaluation_table(no_bmi.models_ups,      split_scaled$df.no_bmi.test)
no_bmi.evaluation_table_downs <- create_evaluation_table(no_bmi.models_downs,    split_scaled$df.no_bmi.test)
write.csv(bmi.evaluation_table,          file.path(RESULT_FOLDER, 'lr_svm.without_feat_eng.csv'))
write.csv(bmi.evaluation_table_ups,      file.path(RESULT_FOLDER, 'lr_svm_upsampled.without_feat_eng.csv'))
write.csv(bmi.evaluation_table_downs,    file.path(RESULT_FOLDER, 'lr_svm_downsampled.without_feat_eng.csv'))

## 2.3. XGBoost
bmi.xgb_default <- train_default_xgb_model(split$df.bmi.train, 'NObeyesdad', model_name='XGB default (with BMI)')
bmi.xgb_default_up <- train_default_xgb_model(split$df.bmi.train_up, 'NObeyesdad', model_name='XGB upsampled (with BMI)')
bmi.xgb_default_down <- train_default_xgb_model(split$df.bmi.train_down, 'NObeyesdad', model_name='XGB downsampled (with BMI)')
no_bmi.xgb_default <- train_default_xgb_model(split$df.no_bmi.train, 'NObeyesdad', model_name='XGB default (no BMI)')
no_bmi.xgb_default_up <- train_default_xgb_model(split$df.no_bmi.train_up, 'NObeyesdad', model_name='XGB upsampled (no BMI)')
no_bmi.xgb_default_down <- train_default_xgb_model(split$df.no_bmi.train_down, 'NObeyesdad', model_name='XGB downsampled (no BMI)')

### Store results
xgboost.eval_table <- list(bmi.xgb_default, bmi.xgb_default_up, bmi.xgb_default_down, no_bmi.xgb_default, no_bmi.xgb_default_up, no_bmi.xgb_default_down)
xgboost.eval_table <- do.call(rbind, lapply(xgboost.eval_table, as.data.frame))
write.csv(xgboost.eval_table, file.path(RESULT_FOLDER, 'xgboost.evaluation_results.csv'))

# 3. Hyper-parameter tuning (For simplicity - only data with BMI considered)
## Tune for random forest
tuneGrid        <- expand.grid(mtry = c(4, 6, 8, 10, 12, 14))
controls        <- trainControl(method = "cv", number = 5, verboseIter = TRUE)
rf_tuned        <- train(NObeyesdad ~ ., 
                         method = "rf", 
                         data = split$df.bmi.train, 
                         tuneGrid = tuneGrid, 
                         trControl = controls, 
                         metric = "Accuracy")
rf_tuned_ups    <- train(NObeyesdad ~ ., 
                         method = "rf", 
                         data = split$df.bmi.train_up, 
                         tuneGrid = tuneGrid, 
                         trControl = controls, 
                         metric = "Accuracy")
rf_tuned_downs  <- train(NObeyesdad ~ ., 
                         method = "rf", 
                         data = split$df.bmi.train_down, 
                         tuneGrid = tuneGrid, 
                         trControl = controls, 
                         metric = "Accuracy")
rf_tuned.models <- list(
  "Random Forest (Tuned)"             = rf_tuned,
  "Random Forest (Tuned, Upsampled)"  = rf_tuned_ups,
  "Random Forest (Tuned, Downsampled" = rf_tuned_downs
)
tuned_rf.evaluation_table <- create_evaluation_table(rf_tuned.models, split$df.bmi.test)
write.csv(tuned_rf.evaluation_table,     file.path(RESULT_FOLDER, 'tuned_random_forest.with_feat_eng.csv'))

