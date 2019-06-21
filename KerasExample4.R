library(tensorflow)
library(keras)


##A linear regression example with shelter inflation against unemployment by city.
##The results are interesting, not great. 


CityData <- read.csv(paste(filefolder,"ConsolidatedData_short.csv",sep=""), stringsAsFactors=FALSE)
CityData <- CityData[CityData$MetricYear>=1990,]
CityData$city <- as.integer(as.factor(CityData$city))

Cities <- data.frame(sort(unique(CityData$city)))
colnames(Cities) <- c("city")
Cities$cmsa <- rep(NA,dim(Cities)[1])
for(i in Cities$city){
 Cities$cmsa[Cities$city==i] <- unique(CityData$cmsa[CityData$city==i])
}
drops <- colnames(CityData) %in% c("cmsa")
CityData <- CityData[,!drops]


indyvars <- c("city","urate")
depvars <- c("SAH1")
setofvars <- c(indyvars,depvars)
for(i in setofvars){
  CityData <- CityData[!is.na( eval(parse(text=paste("CityData$",i,sep=""))) ), ]
}
subsetselection <- CityData$MetricYear <2005
train <- as.matrix(CityData[subsetselection, ])
test <- as.matrix(CityData[!subsetselection, ])



train_data <- train[,indyvars]
train_labels <- train[,depvars]
test_data <- test[,indyvars]
test_labels <- test[,depvars]


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

epochs <- 150  #okay, I shortened the epochs from 500 to 30

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
#class_pred <- model %>% predict_classes(test_data)



##Okay back to my stuff

plot(test_predictions~test_labels)  
abline(0,1)
cor(test_predictions,test_labels)
#I see some misclassified elements. 

for(i in indyvars){
plot(test_predictions~test_data[,i], xlab=i)
readline("next?")
}


plot(test_predictions~test_data[,"urate"], pch=16,col=test_data[,"city"])


#Hmmm so it basically divides up the domain of the unemployment rate and creates a non-linear model.
#The model does not do a great job in sample and out-of-sample it is surprising that it works at all. 
#I am fairly sure it needs more variables, but at this moment I don't know which to give it and I don't know why. 


#I think I would stick with normal linear regression, but this has been insightful. 



#what does the model really think here?
#Lets create a data.frame that saturates the space with points and predict values
extratest <- expand.grid(city=c(1:27), urate=c(-100:1500)/100)
extrapred <- model %>% predict(as.matrix(extratest))
#Now plot it to see 
plot(extrapred~extratest$urate, col=extratest$city)




