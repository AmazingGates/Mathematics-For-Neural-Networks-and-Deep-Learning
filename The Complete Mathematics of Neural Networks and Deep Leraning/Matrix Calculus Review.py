# In this section we will be Reviewing Matrix Calculus.

#   Gradients (Rn ---> R1)

#   Gradients -
# - How we display the partial derivatives of a function that takes something from the vector to being a scalar.
# - Example below.

# Let's say f of x and y, so that's two input variables, equals x squared 2, plus cosine of y.
#   f(x,y) = x2 + cos(y)

# Now let's take the partial derivartives, with respect to both variables.

#   2f
#------- = 2x
#   2x
#
#
#   2f
#-------- = -sin(y)
#   2y
#

# So those are two partial derivatives. 

# Now we will explore what we meant by taking something from a vector to a scalar.

# So let's take this function f(x,y) = x2 + cos(y). We can display (x,y) as a vector and put it through f(x,y),
#and then it return some sort of scalar.

#  __  __
# |      |
# |  X   |
# |      | ------> (s)
# |  Y   | f(x,y)
# |      |
# |__  __|

# To make it clearer let's add some numbers.

# Let's say x = 2, and y = 0

# That will give us this.

#   f(x,y) = x2 + cos(y)

#  __  __
# |      |
# |  2   |
# |      | ------> (s)
# |  0   | f(x,y)
# |      |
# |__  __|

# So 2 squared is going to be 4, and the cosine of 0 is going to be 1.

# So we're going to add those numbers to get our scalar, which is 5.

#   f(x,y) = x squared + cos(y)

# Let's say x = 2, and y = 0

#  __  __
# |      |
# |  2   |
# |      | ------> (5)
# |  0   | f(x,y)   R1
# |      |
# |__  __|
#    R2

# So that's our vector, which is R2, to our scalar, which is R1.

# Now back to the gradient, which is a way we can display a partial derivative of this function, or any function like
#this that takes a vector to a scalar.

# A gradient is just a vector of the partial derivative, which in this case would be (see example below)

#  __   __
# |       |
# |  2f   |
# | ----  | 
# |  2x   | 
# |       |  Vector
# |  2f   |
# | ----  |
# |  2y   |
# |__   __|

# So it would be a partial derivative in respect to x, and a partial derivative in respect to y.

# And that would go on for however many input variables we have. So if we had 1,000 input variables, there would 1,000
#items in our input vector with respect to each input variable. In this case, we only have two.

# And our gradient would equal 2x and -sin(y) (see example below)

#  __   __          ___   ___
# |       |        |         |
# |       |        |         |
# |  2f   |        |   2x    |  
# | ----  |        |         |
# |  2x   |        | -sin(y) |
# |       |   =    |         | Partial Derivative
# |  2f   |        |         |
# | ----  |        |         |
# |  2y   |        |         |
# |__   __|        |___   ___|

# This would be defined as the gradient of f(x,y).



#   Jacobians (Rn ---> Rm)

#   Jacobians -
# - Instead of taking a vector to a sclar, it takes a vector to a vector.
# - Might be of the same, or different shape.
# - Let's see an example below.

# We'll take the same function f(x,y), and instead of returning a scalar, it's going to return a vector.

#               _____   ____
#              |            |
#   f(x,y) =   |  2x - y3   |  (y3 = y to the third power)
# R2 input     |            |
#              |  ex - 13y  |   (ex = e to the power of x, 13y = 13 to the power of y)
#              |_____   ____|
#                 R2 output
# 

# This is our function that takes in an R2 input and returns an R2 output.

# Taking a partial derivative of this is a little more complicated, because now we have more than one function,
#technically.

# So how we're going to break this up, is we're going to make into two scalar functions.

# So we're going to say f1 is equal to the first row, 2x - y3 (y to the third power)
#   f1 = 2x - y3

# And f2 is equal to the second row, ex - 13y (e to the power of x - 13 to the power of y)
#   f2 = ex - 13y

# So now we just take the partial derivative of each of these funtions, with respect to the variables, just like we did
#before, but now we're kinda doing them separately.

# So let's take the first function, f1 = 2x + y3 (y to the third power)

# Now let's take the derivative of f1, with respect to x. 

# 2 f1
#------ = 2
# 2x

# Next will take the derivative of f1 with respect to y.

# 2 f1
#------ = -3y2 (3 times y-squared)
# 2y

# So these are our two derivatives of the first function (f1)

# Now we will do the same for the second function (f2)

# First we have the f2 with respect to x.

# 2 f2
#------ = ex (e to the power of x)
# 2x

# Then we the f2 with respect to y.

# 2 f2
#------ = -13
# 2y

# So now we have four partial derivatives, two for the f1 function, and two for the f2 function.

# But similar to the gradient, how we're going to calculate this Jacobian, is going to be how we kinda assemble 
#these partial derivatives, 

# So let's make the Jacobian.

# All we do is arrange the partial drivatives into our Jacobian.

# Will we draw two lines in our Jacobian for our two sets of derivatives,with respect to our two functions,
#f1, and f2.

# Note: The more lines our output has, means the more lines our Jacobian has.

# Also, another way we can think of this is that each line of the Jacobian is a gradient of each function (transposed).
#(Transposed) because they are no longer displayed vertically, but are now displayed horizontally.

# 

#          ___                   ___________
#         |                                 |
#         | 2 f1        2 f1                |
#         |------      ------ Gradient f1   |
#         |  2x         2y                  |
#    J =  |                                 |
#         | 2 f2        2 f2                |
#         |------      ------ Gradient f2   |
#         | 2x          2y                  |
#         |___                   ___________|

# Ultimately, this is the Jacobian will we have for our original function. (See Below)

#          ____    ____
#         |            |
#         |            |
#         |  2    -3y2 | (-3 times y-squared)
#         |            |
#    J =  |            |
#         |            |
#         | ex    -13  | (e to the power of x)
#         |            |
#         |____    ____|



#   Jacobian Chain Rule

#   Jacobian Chain Rule -
# - Before looking at the Jacobian Chain Rule, We will look at the Scalar Chain Rule in a new way to help us better
#understand the J C R.

# - Scalar Chain Rule will get us familiar with differentiating functions. This will help us differentiate larger 
#functions and get more comfortable with Jacobian Chain Rules.

# - S C R - If we have some function we can use the chain rule on, sin (x2), (x2 = x-squared) , for example, normally 
#what we would dois multiply the derivative of the inner function by the derivative of the outter function, 
#which would get us 2x cos (x2), (x2 = x-squared)

# - Now we will see how that will help us understand the J C R a little better and deal with vector to vector functions.

# - Let's look at the J C R example below

#           ______                  ______
# f(x,y) = | sin(x-squared + y)           |
#          |  ln (y to the third power)   |
#          |______                  ______|

# Looking at this example, we can see that both of these are functions that would generally require the chain rule
#if we were dealing with them by themselves.

# We would multiply the derivative of the outside by the derivative of the inside.

# So we're do something similar to what we've done before.

# We can set an itermidiate variable for the insides, and turn them into vector intermidiate functions.

#      ________       _________
#     | x-squared + y          | (g1)
# g = |                        |
#     |_y to the third power___| (g2)

# These are our two inside internmidiate funtions 

# Now we will set an outer function vector

#      ________       _________
#     |       sin(g1)          |
# f = |                        |
#     |________ln(g2)__________|

# Now that we have our inner and outer intermidiate function vectors set, we can move on from here.

# What we can do is, we can Jacobian both of these.

# We will compute the Jacobian of g first.

# The Jacobian of g is going to be how g changes when we change x and y.

# So to be more specific, that's going to be how g1 and g2 changes when we change x and y.

# So how we can represerent this is the change in g, which is a vector now, of g1 and g2, when we change x, which is 
#a vector now of x and y.

# So we'll take the first function of g1 (x-squared + y), and see how that changes with x.

# x-squared changes to 2x.

# y changes to 1.

# Since there is no x in the g2 function, the blank x column changes to a 0.

# y to the third power changes to 3 times y-squared

# Now we will do the exact samething for the second vector.

# Now we're going to be finding the change in f, when we change g.

# Now we're going to be doing another Jacobian, but this time instead of our variables being x and y, they are going 
#to be g1 and g2, because those are our inputs to the f vector.

# So we'll start with sin(g1) with respect to g1, which is going to change to cosine of g1.

# Next, cosine of g1 with respect to g2 changes to 0.

# Next, lawn of g2 with respect g1 changes to 0.

# And lawn of g2 with respesct to g2 changes to 1 over g2

#   -> = (This symbol, or one similar to it, equals change, or a change in a variable)
#  ________________
# |  ->      _______
# | 2g      | 
# |------ = | 2x  1
# |  ->     |
# | 2x      | 0   3y-squared
# |         |_______
# |
# |
# |  ->      ________
# | 2f      |
# |------ = | cos(g1)  0
# |  ->     |
# | 2g      | 0  _1_
# |         |____ g2__
# |_________________


# Now we have our two Jacobians.

# What can we do with them?

# We might remember with the scalar chain rule, we can find the derivative of some y with respect to some x, if we 
#we have the derivative of that y to some intermidiate u, for example, and that intermidiate u to some x.

#   2y       2y      2u
#------- = ------  ------
#   2x       2u      2x

# We can almost think of them as crossing each other out, leaving a y with respect to x.

# And keeping that in mind, we can actually do the samething with the Jacobian chain rule.

# Now that we're familiar with the J C R, we could see how that comes about pretty easily.

# So right now we're trying to find the change in the f vector, when we change the x vector.

#   ->
#  2f
#------
#   ->
#  2x


#           ______                  ______
# f(x,y) = | sin(x-squared + y)           |
#          |  ln (y to the third power)   | = f vector 
#          |______                  ______|


# x vector is going to be a Jacobian of the f vector.

# That's going to be 

#  ______
# |  ->     
# | 2g      
# |------ 
# |  ->    
# | 2x      
# |         

# Multiplied by

# |  ->    
# | 2f      
# |------ 
# |  ->     
# | 2g      
# |       
# |______

# And with matrix multiplication, it's really important to get the order correct.

# Because if we multiply some matrix A times some matrix B, that's not equal to some matrix B times some matrix A.

# So we need to be really careful about our ordering.

# So what we do when we are multiplying Jacobians for the J C R, we wanna start with the outter functions first (bottom
#vector of the Jacobian Chain Rule). (See lines 315 - 321)

# So we're going to find the change in f when we change our g, and then how g changes when we change our x.

#   ->       ->      ->
#  2f       2f      2g
#------ = ------  ------
#   ->       ->      ->
#  2x       2g      2x

# Now that we have this, we can directly substitute the variables out for the values and see what we get.


# |  ->      _______  _______   _______  ______  
# | 2f      |                | |               |
# |------ = | cos(g1)  0     | | 2x        1   |
# |  ->     |                | |               |
# | 2x      | 0       _1_    | | 0   3y-squared|
# |         |_______   g2  __| |________  _____|
# |    

# Now we can do simple matrix multiplication of these two to get our answer.

# So, what we're trying to do is find the the change in f when we change x. And this is how we are going to
#carry out that process.

# First we multiply the first two rows in both our matrices. 

# cos(g1) times 2x = 2x cos(g1)
# 0 times 2x = 0
# cos(g1) times 1 = cos(g1)
# 0 times 1 = 0


# Next we multiply the second rows in both our matrices.

# 0 times 0 = 0
# 1 over g2 times 0 = 0
# 0 times 3y-squared = 0
# 1 over g2 times 3y-squared = 3y-squared over g2


# |  ->      ___________  ____________
# | 2f      |                         |      
# |------ = | 2x cos(g1)  cos(g1)     |
# |  ->     |                         |
# | 2x      |  0         3y-squared   |
# |         |           ------------- |
# |         |___________  __g2________|
# |    

# And the last step here is going to be substituting the (g1)'s with the actual intermidiate functions.


# |  ->      ___________  __________________________________
# | 2f      |                                               |      
# |------ = | 2x cos(x-squared + y)  cos(x-squared + y)     |
# |  ->     |                                               |
# | 2x      |  0                        3y-squared          |
# |         |                          -------------        |
# |         |_____________  ________y to the third power____|
# |    

# And finally, we can simplify 3y-squared over y to the third power to get our final answer

# |  ->      ________________________  _____________________
# | 2f      |                                               |      
# |------ = | 2x cos(x-squared + y)  cos(x-squared + y)     |
# |  ->     |                                               |
# | 2x      |  0                             3              |
# |         |                          -------------        |
# |         |_______________________________ y _____________|


# This Jacobian is the change in our f when we change our input vector x, giving us our final answer.


# Note: If we ever ran into a function that we are not able to use the J C R on, when we have a function that 
#doesn't have an intermiduate function, which will happen a lot, we will use a 1 for computing purposes, not a 0.