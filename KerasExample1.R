library(tensorflow)
library(keras)


##A simple example where the ML algorithm needs to find a separating hyperplane
## Everything above the line will be a 1, everything below will be a 0. 

n <- 10^4
exes <- runif(n)
whys <- runif(n)
labels <- rep(0,n)
labels[whys>.3 + .7*exes] <- 1
plot(whys ~ exes, col=labels+1)
abline(.3,.7,col='red',lwd=2)

subsetselection <- runif(n)
train <- subsetselection < .9#.5
#validation <- subsetselection >= .5 & subsetselection <.9
test <- subsetselection >= .9

plot(whys ~ exes, col=labels+1, subset=test)




data <- cbind(whys, exes)


train_data <- data[train,]
train_labels <- labels[train]
test_data <- data[test,]
test_labels <- labels[test]


###This is taken from the Keras Boston Housing tutorial

build_model <- function() {
  
  model <- keras_model_sequential() %>%
    layer_dense(units = 64, activation = "relu",
                input_shape = dim(train_data)[2]) %>%
    layer_dense(units = 64, activation = "relu") %>%
    layer_dense(units = 1)
  
  model %>% compile(
    loss = "mse",
    optimizer = optimizer_rmsprop(),
    metrics = list("mean_absolute_error")
  )
  
  model
}

model <- build_model()
model %>% summary()

print_dot_callback <- callback_lambda(
  on_epoch_end = function(epoch, logs) {
    if (epoch %% 80 == 0) cat("\n")
    cat(".")
  }
)    

epochs <- 30  #okay, I shortened the epochs from 500 to 30

# Fit the model and store training stats
history <- model %>% fit(
  train_data,
  train_labels,
  epochs = epochs,
  validation_split = 0.2,
  verbose = 0,
  callbacks = list(print_dot_callback)
)



test_predictions <- model %>% predict(test_data)
class_pred <- model %>% predict_classes(test_data)



##Okay back to my stuff

xtabs(~class_pred+test_labels)  
#I see some misclassified elements. 

#What are the predicted values
hist(test_predictions[class_pred!=test_labels], breaks=100)


#show me where they are in (why,exe) space, and highlight the misclassifications
plot(whys ~ exes, col=labels+1, subset=test)
points(test_data[,1] ~ test_data[,2], col=labels[test]+1, subset=class_pred!=test_labels, pch=16,cex=2)
abline(.3,.7,col='red',lwd=2)
