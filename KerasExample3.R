library(tensorflow)
library(keras)


##Another example where the ML algorithm needs to find a separating hyperplane
## Everything around a set of points will be a 1 elsewhere will be a 0

n <- 10^4
exes <- runif(n)
whys <- runif(n)
labels <- rep(0,n)

thepoint <- t(matrix(c(.65,.8,.3,.25),nrow=2,ncol=2))
error <- .1
these <- thepoint[,1] + error > exes & thepoint[,1] - error < exes & thepoint[,2] + error > whys & thepoint[,2] - error < whys 
labels[these] <- 1

plot(whys ~ exes)
points(thepoint, pch=16,col='purple',cex=2)
points(whys ~ exes, subset=these, col="purple")


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

epochs <- 100  #okay, I shortened the epochs from 500 to 30

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



#what does the model really think here?
#Lets create a data.frame that saturates the space with points and predict values
extratest <- expand.grid(whys=c(1:100)/100, exes=c(1:100)/100)
extrapred_class <- model %>% predict_classes(as.matrix(extratest))

#Now plot it to see 
plot(extratest$whys~extratest$exes)
points(extratest$whys~extratest$exes,col="purple", subset=thepoint[,1] + error > extratest$exes & thepoint[,1] - error < extratest$exes & thepoint[,2] + error > extratest$whys & thepoint[,2] - error < extratest$whys )
points(extratest$whys~extratest$exes,col="red", subset=extrapred_class==TRUE)
points(thepoint, pch=16,col='purple',cex=2)

#Why is the second point not there?!!
##Oh, it is because I only had 30 epochs... that's wild
##The more epochs that I give it to train on, the better.  (I gave it 100). 


##I guess it is the limited domains of the whys and exes in the training sets that explains why only part of the squares are predicted. 

