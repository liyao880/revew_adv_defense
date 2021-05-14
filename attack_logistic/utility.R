
get_zero_one = function(data){
  index = which(data$labels==0|data$labels==1)
  new = list()
  new$images = data$images[index,]
  labels = data$labels[index]
  labels[which(labels==0)]=-1
  new$labels = labels
  return(new)
}

gradient = function(x,y,w){
  tmp = -y/(1+exp(y*(w%*%x)))
  return(as.vector(tmp) * as.vector(w))
}

gradient_CW = function(x,y,w){
  tmp = w%*%x
  if (y*tmp<0){
    grad = 0
  } else {
    grad = as.vector(y) * as.vector(w)
  }
  return(grad)
}

clamp = function(x_new,x,epsilon){
  n = length(x)
  for (i in 1:n){
    x_new[i] = max(min(x[i]+epsilon,x_new[i]),x[i]-epsilon)
  }
  return(x_new)
}

PGD = function(w,x,y,epsilon,alpha,iters){
  x_new = x
  for (i in 1:iters){
    x_new = x_new + alpha * sign(gradient(x_new,y,w))
    x_new = clamp(x_new,x,epsilon)
  }
  return(x_new)
}

attack_over_test = function(test,prediction,classifier,epsilon,alpha,iters){
  w = classifier$W
  corr = 0
  n = length(test$labels)
  attack = rep(0,n)
  for (i in 1:n){
    x = test$images[i,]
    y = test$labels[i]
    if (prediction[i] != y){
      next
    }
    x_new = PGD(w,x,y,epsilon,alpha,iters)
    pred = predict(classifier, matrix(x_new,1,784), proba = FALSE, decisionValues = FALSE)$predictions
    if (pred == y){
      corr = corr + 1
    } else {attack[i] = 1}
  }
  attack = which(attack==1)
  cat("The accuracy under attack is: ",corr/2115)
  return(attack)
}

Visual_adv = function(i,data, prediction,classifier,epsilon,alpha,iters){
  #----Choose a data point. ----#
  x = data$images[i,]
  y = data$labels[i]
  #----Visualize the image. ----#
  par(mfrow=c(1,2))
  image(1:28, 1:28, matrix(x, nrow=28)[ , 28:1], 
        col = gray(seq(0, 1, 0.05)), xlab = "Original Image", ylab="")
  #----Check if the original prediction is correct. ----#
  cat("The true label for this image is: ",y)
  cat("The predicted label for this image is: ", prediction[i])
  if (y != prediction[i]) {
    print("We only proceed when predicted label equals to true label")
    break
  }
  #----Generate adversarial examples. ----#
  w = classifier$W
  x_new = PGD(w,x,y,epsilon,alpha,iters)
  cat("The L2 distortion is: ",sum((x-x_new)**2))
  cat("The L_inf distortion is: ",max(x-x_new))
  #----Visualize adversarial examples. ----#
  image(1:28, 1:28, matrix(x_new, nrow=28)[ , 28:1], 
        col = gray(seq(0, 1, 0.05)), xlab = "Adversarial Image", ylab="")
  #----Check if the attack is successful. ----#
  pred = predict(classifier, matrix(x_new,1,784), proba = FALSE, decisionValues = FALSE)$predictions
  if (pred != y){
    print("The attack is successful!")
  }  
}

adv_train = function(train,prediction,classifier,epsilon,alpha,iters){
  w = classifier$W
  corr = 0
  n = length(train$labels)
  adv_example = train$images
  for (i in 1:n){
    x = train$images[i,]
    y = train$labels[i]
    if (prediction[i] != y){
      next
    }
    x_new = PGD(w,x,y,epsilon,alpha,iters)
    adv_example[i,] = x_new
    pred = predict(classifier, matrix(x_new,1,784), proba = FALSE, decisionValues = FALSE)$predictions
    if (pred == y){
      corr = corr + 1
    }
  }
  cat("The accuracy under attack is: ",corr/n)
  print("Training with adversarial examples")
  return(adv_example)
}