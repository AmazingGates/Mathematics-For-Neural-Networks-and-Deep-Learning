# We will Begin The Course On The subject Of The Complete Mathematics For Neural Networks And Deep Learning

# First we will dive into the topic of notation

#   Notation -
# - m = size of train set
# - n = amount of input variables
# - L = amount of layers in Neural Network
#       - l = Specific layer
# - Wl = Weight(s), l = layer going into
#       - bl = Bias for layer l
# - X1 = Single variable input 


# Here we will look at some Big Picture concepts

#   Neural Networks as Functions -

#   Neural Networks as Functions - It's important to keep in mind that aneural network is just a fancy function,
#made up of a lot of little functions. All the parameters to this function are the weights and the biases,
#however many that may be. And the inputs to this function is usually vector of all the variables of one training
#example. 
# So we have this beginning of a neural network here for example.
# And this all go to whatever layer they will be connected to.
# And the inputs are something like X1, X2, X3 ... Xn

# X1 O -
# X2 O -
# X3 O -
# X4 O -
# X5 O -
# X6 O -

# And the output can be a lot of different things depending on the neural network.
# Often if we're thinking of something like image classification, it might be a vector of probabilities, and 
#we choose the biggest probability for our classification, or something like that, but the output is often a scaler,
#or a decision, or a yes or a no.
# So it's basically turning the raw numbers into something understandable.


#   Just a Big Calculus Problem -

#   Secondly, we want to think of this as just one big calculus problem. That's really all it is. We're trying to
#minimize the cost function. So we might see terminator, or something like that, all those are are just 
#calculus. Really just one big minimization problem in calculus.
# We're trying to take the cost, and trying to minimize that. 


#   All About 2 cost -
#              2 W

#   This is all about the cost function and minimizing that cost function. More specifically, it's all about finding
#the derivative of the cost function with respect to every single weight and bias. So basically we can think about
#it like this, if we have a function with let's say 1,000 parameters, so 1,000 weights and biases, to best figure
#out how to lower that cost function per se, is to find out how much each weight and bias contributes to the cost.
# So then we can appropriately add or subtract the different weights to make sure we're finding our optimal
#algorithm. So basically that's why we're trying to do all of this. And that's why the entire goal of backpropagation 
#is to find the derivative of the cost with respect to every different weight and bias, so basically how the cost
#changes when we change a weight or bias in our algorithm.