# In this section we will be dicussing Derivatives Of Neural Networks


# Here we will begin with going over Motivation and Cost Functions


# The first thing we will look at is back propagation.

# We're going to be seeing how we can lower the cost of our final output, or the accuracy of the error.

# Minimize the error of our output by changing the weights and the biases in our entire algorithm.

# So how we're going to this is by going back through the network and seeing how each of the weights in our network
#effect the total cost.

# So if we tweak a random weight neural network, will that increase or decrease the final cost.

# And using that information, we can tweak those weights and biases accordingly and get a better answer.

# That's basically the entire goal of back propagagtion.

# The first thing we'll cover is the cost function.

# One of the most popular cost functions is the called the Mean Squared Error.

# The Mean Squared Error allows us to cycle through each one of our outputs and then we compare them to our actual real
#answers, and then we square the error, and then we sum over it.

# We are going to start off with a very simple scenario.

# We'll have 5 inputs going into a single neuron.

# That will be a vector turned into a scalar output, which is the activation of a single neuron.

# Then we'll just put that to a cost function

# X1
# X2
# X3   O => C
# X4
# X5

# Using this very simple neural network with a single node, we will explore how changing the five weights, plus the
#bias, is going to change our cost.

# And we're going to explore the idea of using Jacobians and seeing how we can calculate the derivatives of the cost
#function with respect to each of our weights and our bias.

# We will explore the changes our cost goes through when we change each of our derivatives.

# We can use that to better tweak our weights and increase the accuracy and lower the error of our final output.

# We're going to do that by calculating the derivatives.

# We will aslo be going over gradient descent, and how to use the derivatives to create an algorithm to automatically
#find the lowest cost.


# Here, we will be talking about Differentiating a Neurons Operations

# First we will be going over Derivative of A Binary Element-wise Operation

# Now that we've talked about the operations that go on within a single neuron and a layer of neurons, now we're
#going to try to find the derivatives of these different operations so we can go back and try to find the derivatives of an
#entire neuron with respect to the weights and the bias.

# So how do we start to do this?

# First we will be finding the Jacobians/Derivatives of the Binary Element-Wise Operations, and the Hadamard Product.

# The Hadamard Product is actually a Binary Element-Wise Operation, but we will go over why they are separated.

# What is a Binary Element-Wise Operation?

# Binary Element-Wise Operation is some sort of function that takes in two vectors, V and W, and returns a single
#vector, B.

#   -> ->     ->
# f(V, W)---> B

# So, what does this mean?

# The operation that we perform on the V and W needs to be element-wise.

# So we have some vector V, and we have some vector W, and we're doing some element-wise operation between them.

# It has to be element-wise, so it has to be addition, subtraction, multiplication, division, comparison, etc etc...

# So the output of this function needs to be the first element of V, plus the first element of W, and the second
#element of V plus the second element of W will give us the second element in the output Vector.

#   __ __     __ __     __ __
#  |     |   |     |   |     |
#  |     |   |     |   |     |
#  |  V  | + |  W  | = |     |
#  |     |   |     |   |     |
#  |__ __|   |__ __|   |__ __|
#

# This goes for all of the other element-wise operations as well.

# Note that the definition of a Hadamard is a Binary element-wise operation when we're using Multiplication.

# So how can we take the derivative of a binary element-wise operation?

# And obviously it's binary because there two vectors involved.

# First, Let's put it in standard function notation.

# Let's take some (v,w) and then say an explicit notation, f of (v), and then some sort of element-wise operation,
#and then g of (w) 

#   -> ->     ->     ->
# F(V, W) = f(V) + g(W)

# The reason why we have the "f" and "g" there is because sometimes we might want to do something on he Vector before
#comparing it.

# So for example maybe we want to multiply every element in V by 5 before comparing it to W.

# Or maybe to multiply every element in V by 5 and then every element in W by 6.

# Or maybe take the log of every element.

# But the thing is, these functions don't have to be element-wise. 

# The total function needs to be element-wise 
#   -> ->    
# F(V, W)

# The operator needs to be element-wise (+,-,x,/,<,>)

# But these operations don't
#   ->     ->
# f(V) + g(W) 

# So how do we take the Jacobian of this?

#When looking at this     -> ->     ->     ->
#                       F(V, W) = f(V) + g(W)

# When we were looking at the multivariable functions in the beginning, we were passing single vectors into the
#function.

# So maybe some x,y, which is a 2 by 1 vector.

# But now we have two vectors, so what does that mean?

# That means we get two Jacobians out of it.

# One with respects to the elements in V, and one with respects to the elements in W.

# So we're going to end up with two Jacobians.

# This will be all the elements with respect to W

#   2f
# -- -> --
#   2W


# And this will be all the elements with respect to V

#   2f
# -- -> --
#   2V

# Now we will take one of these and find the derivative, for demonstration purposes.

# So how do we approach this and start finding the Jacobian matrix for this function?

# Well do something similar to what we did before.

# First, we'll make a vector.

# And the output of this is going to be some sort of vector. That's because we're taking two vectors, doing some
#operation on them, and then outputting a vector.

# So what would this output look like if we're going to display it as a vector of functions?

# First we'll have some f1 of v, then some element-wise operation, and then some g1 of w.

# And then we'll have some f2 of v, then some element-wise opereation, and then some g2 w.

# We'll do this down until fn of v, then some element-wise opertaion, and then some gn of w.

#            ______  ______
#           |   ->      -> |
#           |f1(V) O g1(W) |
#   -> ->   |              |
# F(V, W) = |   ->      -> |
#           |f2(V) O g2(W) |
#           |              |
#           |   ->      -> |
#           |fn(V) O gn(W) |
#           |______  ______|

# The reason why we don't index the variables v and w, by doing this for example, (V1) or (W3), is because we want 
#the functions, f1 or g1, to be able to access all the elements of the variable, and indexing into the variable may 
#limit the the functions access to a single element, or however many elements specified by the index.

# But we do index the functions, just to organize them better, and to be able to calculate the variables better, with
#respect to each of the functions.

# So what we're going to end up with is some Jacobian.

# We'll use the change in f with a change of v, for example.

# Let's keep in mind that all the rows in the Jacobian are going to be the functions of the vector F(V,W).
#(See lines 187 - 196)

# And all the columns are going to be how that function changes when we change a particular element of V.

# So let's take the first case, F1. 

# So how does F1 (F1 = First row in the vector F(V,W)) change when we change V1. (V1 = the first element in V).
# So this is expressing how a change in V1 effects F1.

# Next we will find the change in V2, and how it effects F1 because we're going in columns, but we're still dealing 
#with F1.

# Lastly for the F1 row, we will find the change in Vn and how it effects F1.

# Now that the first layer/row is done, we can start the process for the second layer/row. (F2)

# We will repest the same steps.

# Finally, we will do all the same steps for the last layer/row, (Fn).

#            ____________________________   _____________________________________
#           |  2      ->      ->    2      ->      ->    2      ->      ->       |
#    2f     |----- f1(V) O g1(W)  ----- f1(V) O g1(W)  ----- f1(V) O g1(W)       |
#  -- -> -- | 2V1                  2V2                  2Vn                      |
#    2V =   |                                                                    |
#           |                                                                    |
#           |  2      ->      ->    2      ->      ->    2      ->      ->       |
#           |----- f2(V) O g2(W)  ----- f2(V) O g2(W)  ----- f2(V) O g2(W)       |
#           | 2V1                  2V2                  2Vn                      |
#           |                                                                    |
#           |                                                                    |
#           |  2      ->      ->    2      ->      ->    2      ->      ->       |
#           |----- fn(V) O gn(W)  ----- fn(V) O gn(W)  ----- fn(V) O gn(W)       |
#           | 2V1                  2V2                  2Vn                      |
#           |_____________________________   ____________________________________|


# So basically here we're saying that for each row in this Jacobian (F1, F2, Fn), we're going to be computing how
#each variable (V1, V2, Vn) changes the function.

# But if we actually look at this, we can notice a pattern.

# For example, let's take f1(V) O g1(W) and how it's effected when we change V1. f1 and g1 are basically just
#indexes letting us know which value in the variable we're dealing with. So in essence, we could have say that we want
#to find the change in (V) O (W) when we change V1. This equation will be accurate, since we are dealing with V1
#and it will return a non zero number.

# But let's look at something like this, when we want to find the change in f2(V) O g2(W) when we change V1.

# This time we are indexing into the second value of the variable by to see if it's effected by a change in V1. This
#time the function f2(V) O g2(W) will be uneffected by the change. This is because we are still dealing with V1, but 
#we are indexing into the second value, as specified by the function f2(V) O g2(W).

# The same thing goes for the function fn(V) O gn(W) when we change V1. Since we are indexing into the n value of the
#function, a change in V1 has no effect on this function.

# The only function we see effected by the change in V1 is the f1(V) O g1(W) function. This is because we are indeed
#indexing into the first value of this variable and the change in V1 does effect it.

# Keep in mind that when the variable (V1, V2, Vn)  has no effect on the function (f1(V) O g1(W), f2(V) O g2(W)
#fn(V) O gn(W)), the element because a zero.

# We can reimagine our Jacobian to look like this with that information in mind.

#            ____________________________   _____________________________________
#           |  2      ->      ->                                                 |
#    2f     |----- f1(V) O g1(W)         0                      0                |
#  -- -> -- | 2V1                                                                |
#    2V =   |                                                                    |
#           |                                                                    |
#           |                       2      ->      ->                            |
#           |        0            ----- f2(V) O g2(W)           0                |
#           |                      2V2                                           |
#           |                                                                    |
#           |                                                                    |
#           |                                            2      ->      ->       |
#           |        0                   0             ----- fn(V) O gn(W)       |
#           |                                           2Vn                      |
#           |_____________________________   ____________________________________|


# Notice that the functions in each row that are effected by the corresponding variable change remain, while all the 
#other functions get changed to an element of zero.

# Also notice that all the functions that are effected move in a diagonal direction.

# Now that we have seen this pattern, the thing that we should make clear is, that for element-wise functions,
#the Jacobian will be diagonal.

# Because we're going to be dealing with only some Vn with some Wn component on each function.

# So if we're taking the derivative to something that's not the Vn or Wn component, there's going to be effect on that
#function.

# So, for any element-wise function, we're going to get a diagonal Jacobian matrix.

# If we think of that computational wise, that's a lot easier to figure out than computing an entire Jacobian matrix,
#because we're just dealing with the diagonals.

# The common way to write this is, diag, which is of course short for diagonal, and the all of our diagonal non zero
#values. See example below.


# diag (  2      ->      ->     2      ->      ->      2      ->      ->  )         
#       ----- f1(V) O g1(W),  ----- f2(V) O g2(W),   ----- fn(V) O gn(W)       
#        2V1                   2V2                    2Vn



# Here we will talking about the derivatives of a Hadamard Product

# Recap, a Hadamard Product, element-wise, multiplies two vectors.

# So how do we take the derivative of this with respects to the elements of either W or V?

# First, let's write out our function, which is some function, capital F, takes some vector V, and some vector W,
#and returns to us some function (f1(V) O g1(W), f2(V) O g2(W), fn(V) O gn(W)).


#            ______  ______
#           |   ->      -> |
#           |f1(V) O g1(W) |
#   -> ->   |              |
# F(V, W) = |   ->      -> |
#           |f2(V) O g2(W) |
#           |              |
#           |   ->      -> |
#           |fn(V) O gn(W) |
#           |______  ______|


# In the Hadamard Product, we have some V, and some W ( V of W ), and we're multiplying them. We don't do anything to 
#the vectors before we multiple them.

# Especially in our case, when we're thinking of Hadamard Products in the context of substituting for the dot product,
#we're kinda just breaking down a Dot Product into a Hadamard Product, the sum, we aren't multiplyng anything because
#we're trying to mimic the Dot Product.

# So in that case we don't really need these little functions (f1, f2, fn, g1, g2, gn), because we don't do anything
#with the vectors, we just take them in.

# But the thing is that before when we did have the little vectors, we never really wrote it down, but we would
#technically be indexing the output of these little functions (f1, f2, fn, g1, g2, gn), because it's still element-wise.

# So even though these little functions (f1, f2, fn, g1, g2, gn) are not element-wise, these two (V, W), still need
#to be element-wise.

# For example, let's say that we're dealing with the n row of an output vector.

# We would normally have something like this fn(V) O gn(W). 

# This is just telling us that we are trying to get the output of (f of V), in the n index, and the output of (g of W), 
#in the n index, and then we're comparing those element-wise.

# Now that we don't have that mini function around it (f1, f2, fn, g1, g2, gn), we just assume that we're dealing 
#with the (Vn) and the (Wn) component of our input vectors, because we don't have that function anymore so we can't
#put our notation on these.

# So we'll see, that we will just end up with this.


# Note: We can now use the multiplication symbol since we know we're dealing with the Hadamard Product.
#            _______  _______
#           |   ->       ->  |
#           |  (V1)  X  (W1) |
#   -> ->   |                |
# F(V, W) = |   ->       ->  |
#           |  (V2)  X  (W2) |
#           |                |
#           |   ->       ->  |
#           |  (Vn)  X  (Wn) |
#           |_______  _______|


# So now lookng at our Jacobian, we'll have something like this.


#            ________________   ________________
#           | 2F1        2F1        2F1         |
#    2f     |-----      -----      -----        |
#  -- -> -- | 2V1        2V2        2Vn         |
#    2V =   |                                   |
#           |                                   |
#           | 2F2        2F2        2F2         |
#           |-----      -----      -----        |
#           | 2V1        2V2        2Vn         |
#           |                                   |
#           |                                   |
#           | 2Fn        2Fn        2Fn         |
#           |-----      -----      -----        |
#           | 2V1        2V2        2Vn         |
#           |_________________   _______________|


# So for each of these, how does the first funtion ((V1)  X  (W1) change with respect to V1.

# It's going to change by a factor of (W1), because we're gonna just be multiplying.

# We're going to be looking at a multiplication of the first element of V, and the first element of W.

# The same goes for the second element and the n element.

# So, they're just going to be multiplications, and when we're differentiating multiplications and we have a
#differential with respect to x, and we have some xw, the respect to x is going to be w.

# The same rule apples to our Jacobian.

# Once we do the multiplications to the functions  ((V1)  X  (W1), (V2)  X  (W2), (Vn)  X  (Wn)), with respect to
#(V), this is what we'll have.


#            ________________________   _______________________
#           |                                                  |
#    2f     |       W1            0            0               |
#  -- -> -- |                                                  |
#    2V =   |                                                  |
#           |                                                  |
#           |                                                  |
#           |       0             W2           0               |
#           |                                                  |
#           |                                                  |
#           |                                                  |
#           |                                                  |
#           |       0             0            Wn              |
#           |________________________   _______________________|


# It will be the samething around if we took with respect to (W), but this time we'll have V's instead.


#            ________________________   _______________________
#           |                                                  |
#    2f     |       V1            0            0               |
#  -- -> -- |                                                  |
#    2W =   |                                                  |
#           |                                                  |
#           |                                                  |
#           |       0             V2           0               |
#           |                                                  |
#           |                                                  |
#           |                                                  |
#           |                                                  |
#           |       0             0            Vn              |
#           |________________________   _______________________|




# Here we will be going over the Derivative of A Scalar Expansion

# The Scalar Expansion is basically the derivative of multiplying a scalar by a vector.

# More in depth, it's when we multilple a sum by a vector.

# The results should be a multiplied vector. See example below.


#    __ __     __ __
#   | V1  |   | 2V1 |
#   |     |   |     |
# 2 | V2  | = | 2V2 |
#   |     |   |     |
#   | Vn  |   | 2Vn |
#   |__ __|   |__ __|

# Basically we just multilple every element by the sum.

# Note: It's the same process if the operation was addition, subtraction, division, or any other opertaion.

# The process can be called broardcasting.

# When we broadcast a scalar, we get a scalar expanse. 

# We call it a scalar expansion because we expanded it to be a vector of the same size. See example below


#    __ __     __ __     __ __
#   |  2  |   |  V1 |   | 2V1 |
#   |     |   |     |   |     |
#   |  2  |   |  V2 | = | 2V2 |
#   |     |   |     |   |     |
#   |  2  |   |  Vn |   | 2Vn |
#   |__ __|   |__ __|   |__ __|

# This is basically the visual representation of the process from our original expression.

# So what's the derivative of something like this, a scalar expansion?

# Or, more precise, the scalar expansion and the execuetion.

# So what's the derivative with respect to V?

# Also, what is the derivative of the z (z equals the number we multiple our vector by) that we operate on each 
#element with.

# We can do this in simplier terms.

# Remember before we had some F of v,w (F(v,w)). 

# Now we will be taking only one vector, with some scalar x. (F(V,x))

# So our output is some Function (F), per vector (V), and then some element-wise operation, multiplication in this 
#case, and then g of our scalar (x). In the example below, 2 is our scalar.
# F(V,2) = F(V) x g(2)

# The function g of (x), equals the 1's vector, times our scalar(x)
# g(2) = 1 x 2

# Basically the g of (x) is expanding x into a vector. 

# By multiplying something by the 1's vector, we get a vector of whatever that scalar was. See Example Below


#    __ __     __ __
#   |  1  |   |  2  |
#   |     |   |     |
# 2 |  1  | = |  2  |
#   |     |   |     |
#   |  1  |   |  2  |
#   |__ __|   |__ __|


# With all that being said, we can represent F(V,x) as a vector function.


#           ______   ______
#          |               |
#          | f1(V) x g1(x) |
#          |               |
# F(V,x) = | f2(V) x g2(x) |
#          |               |
#          | fn(V) x gn(x) |
#          |______   ______|

# This is our function and output.

# First, we can write out what the Jacobian is with the respects to the elements of V. 

# This is our Jacobian Matrix

#            ________________   ________________
#           | 2F1        2F1        2F1         |
#    2f     |-----      -----      -----        |
#  -- -> -- | 2V1        2V2        2Vn         |
#    2V =   |                                   |
#           |                                   |
#           | 2F2        2F2        2F2         |
#           |-----      -----      -----        |
#           | 2V1        2V2        2Vn         |
#           |                                   |
#           |                                   |
#           | 2Fn        2Fn        2Fn         |
#           |-----      -----      -----        |
#           | 2V1        2V2        2Vn         |
#           |_________________   _______________|

# How does the first element inour vector 1 change the output of f1.

# It's basically the same as the operation as we performed before where there will be some sort of non zero.

# We will end up with diagonal matrix like we did previously once we are done operating on our Jacobian.

# But times our diagonal will be made up of our scalar(x).

# That was the derivative in respect to V.

# Now we will look at the derivative with respect to x.

# x is unique in the sense that x is a scalar.

# x doesn't have any indexes because it's just a single number.

# So all we can do is see how this scalar changes things in each of our functions (f1, f2, fn).

# This is because we can't move horizontally(along the columns), with the x because this would indicate that we 
#are indexing into a vector, which would be (V1, V2, Vn), but x doesn't have any indexes, so all we can do is go down
#and see how x changes the different functions.

# So when we're taking a derivative with respect to x what we're going to get is a gradient and not a Jacobian,
#because we're going to get something diagonal. 

# And when we think about it this makes sense, because when we're doing a gradient we have something that inputs a 
#vector that outputs a scalar. 

# And now essentially that's what we're doing by taking the derivative with respect to scalar(x), because we're
#disregarding how V effects it.

# So we're going to get some sort of gradient of F with respect to x.

# So we're going to get how f1 changes with respect to x, how f2 changes with respect to x, and all the way down to 
#how fn changes with respect to x.


#         __ __
#        | 2f1 |
# \      |-----|
#  \Fx = | 2x  |
#        |     |
#        | 2f2 |
#        |-----|
#        | 2x  |
#        |     |
#        | 2fn |
#        |-----|
#        | 2x  |
#        |__ __|


# And in this case if we continue with our multiplication this will just be V1, V2, and Vn, just by using simple scalar
#calculus rules.


#         __ __
#        |     |
# \      | V1  |
#  \Fx = |     |
#        |     |
#        |     |
#        | V2  |
#        |     |
#        |     |
#        |     |
#        | Vn  |
#        |__ __|


# So the gradient with respect to x is just going to be a vector of V's, so it's actually going to be equivalent to V.

# This is our scalar expansion.



# Here we will talk about the derivative of a Sum.

# So now we're done with the hard part, which was differentiating a Hadamard Product, or a Binary Element-Wise Function.

# Now we will be looking a something a little more easy. 

# We're differentiating a Sum.

# So what is the function we're trying to differentiate here?

# It's something that takes a vector V, and sums over the elements, all the way up to n.

# So really what our function looks like is the summation up to n, starting at 1, and then some g of V

#        __ __
#       |  V1 |
# ->    |     |
# V  =  |  V2 |          n
#       |     |  =      ____
#       |  V3 |        \        gi(V) 
#       |     |        /____
#       |  Vn |         L = 1
#       |__ __|        

# So the reason why we use g again, is pretty mmuch similar to our element-wise product, it's that we might want to do 
#something to each value of V that might not necessarily be element-wise, that's why we're not indexing the V, we're
#indexing the function.

# An almost better way to show this would be something like this.


#        __ __
#       |  V1 |
# ->    |     |
# V  =  |  V2 |          n
#       |     |  =      ____
#       |  V3 |        \        (g (V))i 
#       |     |        /____
#       |  Vn |         L = 1
#       |__ __|        

# So we might want to think about it like that.

# So example of a g would be maybe we want to multiple each element by 2 before we add it.

# so, let's say that's our g, and just multiple by 2.

#        __ __
#       |  V1 |
# ->    |     |
# V  =  |  V2 |          n
#       |     |  =      ____
#       |  V3 |        \        (2 (V))i 
#       |     |        /____
#       |  Vn |         L = 1
#       |__ __|

# This will return us a vector like this where each element gets multiplied by 2. 

# And then we index into the i value of this new vector.

#        __ __
#       | 2V1 |
# ->    |     |
# V  =  | 2V2 |          n
#       |     |  =      ____
#       | 2V3 |        \        (2 (V))i 
#       |     |        /____
#       | 2Vn |         L = 1
#       |__ __|

# So what is the derivative of this summation function with respect to each of the elements.

# So how does each element in the vector change the overall summation?

# So now with Jacobians, we used to have things where we have multiple functions. So each row is a funcntion.

# But in a summation we only have one, we have the summation.

# So now we're gonna have one row.

# And when we're thinking about it, a summation is a vector to a scalar function.

# We're taking a vector and we're summing over it and getting a scalar.

# So it makes sense that it's something like a gradient, not a Jacobian.

# What we're going to do is, we're going to have a single function, which is the summation (s)

# [-------------------] s

# And each element inside is going to be the summation with respect to a variable in the vector.

# So let's just write this out.

# So we'll have the derivative of s with respect to some V1, the derivative of s with some respect to some V2, all
#the way up until Vn.

#   _________      _________
#  |                        |
#  | 2s      2s      2s     |
#  |-----   -----   -----   |
#  | 2V1     2V2     2Vn    |
#  |_________      _________|

# So expanding this, expanding the s, what we get is something like this.

#   _________________________________   _________________________________      
#  |                                                                     |
#  |  2      ___              2      ___              2      ___         |
#  |-----   \     gi (v)    -----   \     gi (v)    -----   \     gi (v) |
#  | 2V1    /___             2V2    /___             2Vn    /___         |
#  |_________________________________   _________________________________|

# Note: We can't leave g without an index - The summation in each element loops over all functions g 

# gi is always going to be the same, because we're in one single row.

# From here we can the rule that derivative of of a sum is the same as the sum of the derivative, so we can swap around
#the order and have something like this with the summation first, then the derivative.

#   _________________________________   _________________________________      
#  |                                                                     |
#  | ___    2              ___    2            ___    2                  |
#  | \    ----- gi (v)    \     ----- gi (v)   \    ----- gi (v)         |
#  | /___  2V1            /___   2V2           /___  2Vn                 |
#  |_________________________________   _________________________________|

# So now that we have this, we'll consider the case where g is constant, g is nothing, meaning g of v equals v, so 
#there's no change.

# So just a pure summation.

# In that case we can remove the g's from the vector like in the example below.

# Now we'll be indexing V directly because we won't be indexing those functions anymore.

#   _________________________________   _________________________________      
#  |                                                                     |
#  | ___    2              ___    2            ___    2                  |
#  | \    -----  (Vi)      \     -----  (Vi)     \    -----  (Vi)        |
#  | /___  2V1            /___   2V2           /___  2Vn                 |
#  |_________________________________   _________________________________|

# So now what we're going to have is that, the derivative is going to be 0 on each summation until the i in (Vi) equals 1,
#then it will be 1.

# So basically in this summation, we're going to go through i equals 1.

# When V is 1, (Vi) is going to be 1.

# So the derivative of V1 with respect to V1 is 1.

# Then it will become i equals 2, (i for the summation we're in, and 2 for the other two summations in the vector) which 
#will give us 1 + 0 + 0, all the way until n.

# So the result of the first summation is 1.

# For the next summation, the V2 summation, when i equals 1, we will get a 0, but when i equals 2 we will get a 2. And
#V2 with respect to V2 will give us 1. 

# This will become 0 + 1 + 0, up until n.

# So the result of V2 is 1.

# By now we might be seeing a pattern. The entry of this gradient is going to be all 1's. This is because we are 
#generating a 1 for each summation in our gradient.

# And if we have some g, for example, some g that multiplies, so some Z, that multiplies each thing by Z, well 
#then it's going to be Z1 + Z times 0 + Z times 0 up until n (Z1 + Z0 + Z0) for example.

# So if we have some function g that multiplies each thing by some scalar Z, then our output would look like this,
#(see example below)

# [ z, z, z] 

# So we could see that outputs of a summation are usually, if we don't have any g, are just the 1's vector, transposed.

# Or it might be some Z, times the trnsposed 1's vector.

# So that's the derivative of the sum.



#   Derivative of a nueron's activation.

# Here we will be going over the Derivative of a nueron's activation.

# We will be taking the derivative of nuerons with respect to the weight and the bias.

# So let's look at an example.

#   X1
#   X2
#   X3      N
#   X4
#   X5

# Right now, all we're gonna do is focus on finding how the activation of this one node changes when we change the
#weights and the one bias of the system.

# So there's 6 things we can tweak. 

# And then we're trying to find basically the activation with respect to the weights and the bias.

# So let's write out the notation of the single neuron and how it processes these things.

# It's going to be something like this, W transpose X + b, and whatever the letter of activation is equal to.

#  a = r(WtX + b)

# This is something we can use the chain rule on, because we have our activation function plus the function inside.

# So if we trying to take the derivative with respect to W, or B, we can't just go straight from a to W, we need to
#deal with the intermidiate, which is the activation function.

# So we do some type of chain rule, investigative work, where we can say the change in a with respect to W, (we also
#want to do with respect to B as well), is equal to the change in a with respect to Z., and how does Z change when 
#we change the W. (We do the samething for the bias)

# Z becuase z = WtX + b

# This works because 

#  2a        2a     2Z
#------ =  ------ ------
#  2W        2Z     2a
#
#  2a        2a     2Z
#------ =  ------ -------
#  2b        2Z     2b

# So this is what we're looking for.

# So how can we use what we learned before to differentiate this.

# Remember how we separeted the dot product into a hadamard product and a sum?

# So what we can do now is rewrite our equation (a = r(WtX + b)) as a new function.

# We're going to write sum of W, hadamard product of x, plus the bias a = r(sum(W(x)X) + b).

# This is exactly equivalent

# So now we have a couple more layers in our chain rule, because now we have another function

# So let's say that this inner function (W(x)X) is going to be H, and this (sum(W(x)X) is going to be S of H
#S(H).

# So now it's going to be how does Z change when we change this sum.

# And how does the sum change when we change this H.

# And how does H change when we change our W

# The equation for the bias will remain the same. This is because the bias is completely independent of the H and the
#S(H).


#  2a        2a     2Z     2s     2H
#------ =  ------ ------ ------ ------
#  2W        2Z     2s     2H     2W
#
#  2a        2a     2Z
#------ =  ------ -------
#  2b        2Z     2b

# So these are the two chain rules we use, things that we need to figure out.

# In general, how does the hadamard product change when we change one of the operand?

# Remember our Jacobian is just a diagonal matrix of the elements of the other vectors, diag[X1, X2, Xn], for example.

# That's going to be what this is 2H 
#                               ------
#                                 2W

# We can actually label it like this  ______   ______
#                                    |               |
#                                    |  X1  0   0    |
#                          2H        |               |
#                        ------  =   |   0  X2  0    |
#                          2W        |               |
#                                    |   0   0  Xn   |
#                                    |______   ______|

# That can simplified to a vector like this diag[X1, X2, Xn]


# So now, how does the sum change when we change H?

# We're going to be summing over all the elements in the diag vector.

# Since the sum isn't anything fancy, we're not multiplying anything, (remember when we had the g that's just a 
#cost of function) so it's just going to be 1 tranposed (1T), because that's going to be the output derivative of
#our sum when we don't have a g.

# The derivative of S with respect to H is the horizontal 1's vector (1T).

# So we can replace 2s  , with 1T
#                 ------
#                   2H

# So now our equation looks like this so far.


#  2a        2a     2Z     
#------ =  ------ ------  1T   diag[X1, X2, X3]
#  2W        2Z     2s     

# So now we have to worry about how the Z changes when we change s.

# This isn't going to be difficult because is S is just S of H, S(H) plus b, which is 1.

# This is because we are trying to find the S(H) with respect to the activation.

# Now our equation looks a little different.

#  2a        2a       
#------ =  ------   1   1T   diag[X1, X2, Xn]
#  2W        2Z          

# Now we have one more thing left to do.

# Before we do that let's multiple out what we have so far. 

# The 1 transposed (1T) gets multiplied by the  diag vector.

# This will give us a horizontal X vector, or a transposed vector of X [X]T. 

# This is because anything times 1 is itself.

# We can also do the samething with the next 1 in the equation and it will not change our output.

# We still have a transposed vector  of X [X]T

# This is what our new equation looks like we've worked out those problems.

#  2a        2a       
#------ =  ------   [x]T
#  2W        2Z  

# So now we're left dealing with the activation function.

# Keep in mind that there are a lot of different activation functions that we will come across, but for this example
#we will be using the ReLu.

# ReLu is the max of 0 or Z ( max(0,Z) ). 

# Remember that this (WtX + b) is our Z.

# Let's look at a graph for an example


#                      max(0,Z)

#                   Relu(Z)
#                     |                /
#                     |              /
#                     |            /
#                     |          /
#                     |        /
#                     |      /
#                     |    /
#                     |  /
#   __________________|/______________________Z

# Basicaly looking at the graph where we have Z, and the output of ReLu of Z ( ReLu(Z) ).

# Notice our line that come out between the two points, that's our slope.

# The slope of ReLu at Z > 0 is exactly 1.

# Basically, any Z that is less than 0, gets clipped to 0.

# It doesn't stay negative, it just gets clipped to 0.

# And anything that's not negative, so anything above 0, is just itself.

# It's derivative will stay the same, so just 1 times its derivative up until that point, whatever that may be.

# One last thing to note is that the graph of ReLu is a discontinous piecewise function, meaning it will be undefined
#at the point of curvature.

# So how can we take the derivative of something like this, a max(0,Z)

# What we do is, we actually take a piecewise functional.

# So we make two scenarios.

# If Z is less than or equal to 0
#   [Z <= 0]
#
#If z is greater than or equal to 0
#   [Z >= 0]

# What do we do now?

# If Z is less than 0 or equal to 0, then it's going to 0 out everything.

#   _______________
#  |
#  |  0     If z is greater than or equal to 0
#  |
#  |
#  |
#  |
#  |________________

# Our entire derivative is going to become 0.

# Just think of our chain rule.

# This,  2a ,   will be 0.
#      ------ 
#        2Z

# So far our new equation will look like this.

#  2a              
#------ =   0  [x]T
#  2W         

# So now we can get rid of everything else.

# First, our[x]T represents how Z changes with respect to W   2Z
#                                                           ------
#                                                             2W

# Up until that point it's going to be 0 times 2Z
#                                            ------
#                                              2W

# But if Z is greater than 0, it's going to be 1, and that keeps it the same.


#   _______________
#  |        2Z
#  | 0T x ------    If z is greater than or equal to 0
#  |        2W
#  |
#  |        2Z
#  |  1 x ------    If z is greater than or equal to 0
#  |        2W
#  |________________

# Now we can do some substitutions.

# 0 times 2Z over 2W will give us a vector of 0's

# 1 times 2Z over 2W will not change our vector. 

# Remember, 2Z over 2W is represented by [X]T

# Now we can also substitute our Z.

#            _______________
#           |       
#           |  [0]T   if WtX + b <= 0
#  2aL      |
# ----  =   |        
#  2W       |  [X]T  if WtX + b >= 0
#           |        
#           |________________

# That's it. That's how our activation is changed with respect to W.

# For some specific W, this gives the drivative of the activation with respect to W.

# This will end up being a gradient.

# Now that we have that we just have to work on the bias.

# Our bias is just these two terms.

#  2a        2a     2Z
#------ =  ------ -------
#  2b        2Z     2b

# It's how Z changes with respect to b, and if we remember what this looks like, we'll remember that this is our
#Z ( sum(w(x) x) + b ).

# So how our Z changes with respect to b, which is 1, because this  sum(w(x)x)      cancels out to 0
#                                                                  ------------
#                                                                        Z

# So it will be 0 plus bias, whiich is 1, so we'll have 0 + 1 = 1.

# So the first part  2Z   is going to be 1
#                  -------
#                    2b

# And then we just use te same piecewise for 2a     
#                                          ------ 
#                                            2Z     

# Now we're finding the change in a with respect to Z.

# So it'll be this

#            _______________
#           |       
#           |   0 x 1   if WtX + b <= 0
#  2a       |
# ----  =   |        
#  2b       |   1  if WtX + b >= 0       
#           |________________

# Note: This time 0 and 1 are both scalars



#       Derivative Of The Cost For A Simple Neural Network

# Here we will be going over the derivative of cost.

# Now that we're moving on, we're not just dealing with the derivatives of the activation with respect to the
#weights and the biases, but now we're dealing with the cost, with respect to the weights and the biases.

# So continuing with the example, now we're going to have just one more derivative in our chain of derivatives.

#   X1
#     \
#   X2
#     \
#   X3  -- N --> a --> C
#     /
#   X4
#     /
#   X5

# Now after the activation we can compare the activation against the total cost using the mean squared error function,
#which we'll go over later.

#   ________________________________
#  |                  m 
#  |         1       ___
#  | MSE = -----    \      y (actual answer) - activ (last layer of activation, also known as yhat, or predicted)
#  |         2m     /___
#  |                 i=1
#  |__________________________________

# We calculated in the last part how the activation changes with respect to the weight and the bias, so we're already
#more than half way done, because if we think about it we have this chain of derivatives that we use the chain rule to
#compute, and we just have to add one more link to that chain, which is that link between a and c.

# Our goal is to find how c changes with respect to W, as well as how c changes with respect to b.

# Since we've already done most of the work in the chain, all we need to do is work out the final cost derivative that 
#we added.

# That will look like this.

# Note: On order of derivatives in chain rule multiplication - doesn't matter with scalars, but with Jacobians we 
#start with the outer function on the left.

#    2c       2c     2a
#  ------ = ------ ------
#    2W       2a     2W

# So this is the only thing that we don't know and we need to find out, the change in c with respect to a.

# Looking at our MSE, this means that we have to take the derivative of our MSE with respect to our activation.

# And similarly with the bias, we have to figure out how c changes with respect to a in exactly the same way.

#    2c       2c     2a
#  ------ = ------ ------
#    2b       2a     2b

# So basically we have to figure that one derivative, how c changes with respect to a.

# One last thing we need to look at before we move forward is how we express our training examples.

# How we're going to represent this, before we had just one traininig example, and that was a vector, that goes from
#X1 up until Xn.

#   ___ ___
#  |  X1   |
#  |  X2   |
#  |  Xn   |
#  |___ ___|

# Now we're just going to expand this into a matirx where each column is a new training example.

# We can specifiy the training example by using the superscript (top number).

# And we'll specify the variable by using the subscript (bottom number).

#   _________ _________
#  |   1     2     3   |
#  |  X1    X1    X1   |
#  |                   |
#  |   1     2     3   |
#  |  X2    X2    X2   |
#  |                   |
#  |   1     2     3   |
#  |  Xn    Xn    Xn   |
#  |_________ _________|

# This matrix with it's training examples as columns and variables as rows, we will specify with just a Capital X.

# So basically,


#   _________ _________
#  |   1     2     3   |
#  |  X1    X1    X1   |
#  |                   |
#  |   1     2     3   |
#  |  X2    X2    X2   |  =  X
#  |                   |
#  |   1     2     3   |
#  |  Xn    Xn    Xn   |
#  |_________ _________|

# Now we will be working with our cost function

#
#              m
#     1       ___
#   -----    \      y
#     2m     /___
#             i=1


# So each column of our X associates with a right answer y.

# So each of our trainning examples as a right answer, minus the activation, which we'll say is aL, which is the output
#of our neural network that were going to be comparing against our true answer, and then we square that.


#
#              m
#     1       ___
#   -----    \      (y - aL)squared
#     2m     /___
#             i=1

# So let's take some derivatives.

# Is there anything that we can simplify before hand?

# Remember with the cahian rule that we're always trying to find intermediate functions to try and simplify things,
#well this y - aL, is a good candidate for something we can simplify.

# Let's say that y - aL equals v. ( (y - aL) = v )

# Really all we're doing is, we want to take the derivative of all of this with respect to the weight, since that 
#is the nature of the chain rule, we can't go half way, we want to take this y - aL, with respect to the weight.

# So if we're going to make this y - aL, equal to v, we're going to kind of work our way outwards, closer to the cost.

# since we already have 2a  , we just need to find something in between this and the cost with respect to a.
#                     ------
#                       2W

# Since assigning v, we're adding another link in our chain.

# Basically what we're doing , we're going to see how a changes with w, then we're see how v changes when we change
#a, then we're going to make it how c changes when we change v.

#   2c     2V     2a
# ------ ------ ------
#   2V     2a     2W

# So we can see that we adding a new link in our chain by adding the variable V, which represnts (y -aL).

# So now we see that this (y - aL)squared is just ( V )squared.

# So we're trying to find the derivative of c with respect to w with this new link in our chain.

#   2c       2c     2V     2a
# ------ = ------ ------ ------
#   2W       2V     2a     2W

# Since we already have this 2a  , we can just collaspe this into how V changes with w
#                          ------
#                            2w

#   2c       2c     2V     
# ------ = ------ ------ 
#   2W       2V     2W 

# So that means we need to find the derivative of this V ( y - aL),with respect to W.

# So we're going to take the derivative of this ( y - aL ) with respect to W

#   2 ( y - aL )
#  -------------
#       W

# First, we'll deal with the (y), which becomes a zero vector (0).

# And then we have mius aL (-aL).

# And using the chain rule, the -aL with respesct to W is going to be -a over W.

#   -2aL
#  ------
#    2W

# So bassically, the change in V with respect to a is equal to minus aL with respect to W.

#   2V        -2aL
# ------  =  ------
#   2a         2W

# So this in return will end up being how V changes with respect to W equals how -aL changes with respect to W.

#   2V      -2aL
# ------ = ------
#   2w       2W

# So now all we're trying to find is how c changes with respect to V.

#   2c
# ------
#   2V

# So what does this look like?

# First we're going to see how it goes with W still.

#
#                      m
#   2         1       ___
# ------   ------    \      (V)squared
#   2W        2m     /___
#                     i=1

# So let's go to the next step.

# We know that a derivative of a sum is the same as the sum of the derivative.

# Which would look like this.

#
#              m
#     1       ___    2
#   -----    \     ------ (V)squared
#     2m     /___    2W
#             i=1

# Now we're going to take the derivative of that.

# Since we already have this 2W, so 2W of (V)squared is just going to get another chanin, a super insignificate chain,
#because we're just using the power of rule once.

# We see that we have what  2V  is. 
#                         ------
#                           2W

# So to find what  2Vsquared is, is just another chain chain rule.
#                ------
#                  2W

# We need to find the derivative (V)squared, which is just 2V.

# So basically we need to find the chnage in 2(V)squared with respect to 2V.

#    2Vsquared
#  -------------
#       2V

# To make things more clear we split up     2Vsquared       into    2Vsquared          2V
#                                        ---------------          --------------  x  ------
#                                              2W                       2V             2W

# We already know how V changes with respect to W, but (V)squared is another function outside of that so we have
#to take the outer function, which would be how (V)squared changes with resoect to V, and how V  changes with
#respect to W.

# This is how we'll start the process.

#
#              m
#     1       ___     2V      2Vsquared
#   -----    \      ------ . -----------
#     2m     /___     2W         2V
#             i=1

# The 2V numerator and the 2V denominator will cancel each other out and leave us with the new equation below.

# Notice the (V)squared gets simplified into two V's.

#
#              m
#     1       ___       2V
#   -----    \     2V ------
#     2m     /___       2W
#             i=1

# The next thing we can do is remove the 2 from 2m and from te single 2V to give us this new equation.

#
#              m
#     1       ___       2V
#   -----    \      V ------
#     m      /___       2W
#             i=1

# Note: We should remember that we previously learned that  2V      -2aL , which is a piecewise function
#                                                         ------ = ------
#                                                           2W       2W

# Next we will replace the  2V with its piecewise equivalent, which we worked out earlier.
#                         ------
#                           2W

# Remember that our piecewise function is negative. When taking a negatiive of a piecewise function all we do is
#make the options negative.

#                      ______
#              m      |
#     1       ___     | - [0]T   if WtX + b <= 0 
#   -----    \      V |
#     m      /___     | -  2Z    if WtX + b >= 0
#             i=1     |  ------
#                     |___ 2W

# Now that we have this we can actually multiple V into our piecewise function.

# So V times  [0]T, is just gonin to be [0]T.

# And V times -2Z   is going to be     2Z
#            ------               -V ------
#              2W                      2W

# So this will be our new equation

#                      _____
#              m      |
#     1       ___     |      [0]T   if WtX + b <= 0 
#   -----    \        |
#     m      /___     |       2Z    if WtX + b >= 0
#             i=1     | -V  ------
#                     |_____  2W

# So all we do now is substitute.

# First we will substitute the 2Z   with its [X]T
#                            ------
#                              2W

# That will give us.

#                      _____
#              m      |
#     1       ___     |    [0]T   if WtX + b <= 0 
#   -----    \        |
#     m      /___     | -V [X]T   2Z    if WtX + b >= 0
#             i=1     |_____

# Next we should remember that we assigned V to ( y - aL ), so we will have to expand out the V.

# That we bring us here.

#                      _____
#              m      |
#     1       ___     |    [0]T             if WtX + b <= 0 
#   -----    \        |
#     m      /___     | -( y - aL ) [X]T    if WtX + b >= 0
#             i=1     |_____


# Next we can substitute the activation function, which is represented by aL.

# Remember that we are using the ReLU activation function and the entire thing looks like this, r(WtX + b).

# What the ReLU is really, is just the max of 0 or W transposed times X plus b, which we can write like this,
#max(0,WtX + b), keeping in mind that (WtX + b) = Z.

# So what we can do is try and replacing it just to see what happens.


#                      _____
#              m      |
#     1       ___     |    [0]T                          if WtX + b <= 0 
#   -----    \        |
#     m      /___     | -( y - (max(0,WtX + b))) [X]T    if WtX + b >= 0
#             i=1     |_____

# Note: The X transposed ([X]T) is by itself and isn't affected by the substitution.

# So when we look at this, taking the max of 0 or W transposed times X plus b, if WtX + b is less than zero it would
#have been handled by the piecewise function (if WtX + b <= 0), which indicates a vector of zero transposed [0]T,
#so there is really no reason to have the max there because if it chose this option ( -( y - (max(0,WtX + b))) [X]T ),
#WtX + b is greater than zero, so the max becomes redundant, because the piecewise function ( if WtX + b >= 0 ) 
#does the same thing, so we can get rid of the max and the 0 and just take it to mean that -( y - (WtX + b)) [X]T
#is already greater than 0.

# So we can just use that expression in our new piecewise function

#                      _____
#              m      |
#     1       ___     |    [0]T                   if WtX + b <= 0 
#   -----    \        |
#     m      /___     | -( y - (WtX + b)) [X]T    if WtX + b >= 0
#             i=1     |_____

# So we've simplified the line.

# Now we can do the final task of carrying the minus sign over, and we can do that with these steps.

# This, -( y - (WtX + b)) [X]T, gets changed into -y + (WtX + b)) [X]T because we carry the minus sign over, and two
#negativess make a positive.

# We can then rearrange this, -y + (WtX + b)) [X]T, into this, (WtX + b - y) [X]T

# We can now use this new equation in our piecewise function.

#                      _____
#              m      |
#     1       ___     |    [0]T                  if WtX + b <= 0 
#   -----    \        |
#     m      /___     |    (WtX + b - y) [X]T    if WtX + b >= 0
#             i=1     |_____

# And one last thing we can do is we can bring the summation inside the piecewise function.

#                 _____
#                |
#     1          |    [0]T                         if WtX + b <= 0 
#   -----        |
#     m          |    m
#                |  ______
#                | \         (WtX + b - y) [X]T    if WtX + b >= 0
#                | /______
#                |    i=1 
#                |_____


# So that's the derivation of the cost function with respect to the weights.



#       Understanding The Derivative Of The Cost With Respect To The Weights

# At the end of the day, once we're done deriving the change in C when we change some weight W   2c ,
#                                                                                              ------
#                                                                                                2W
#what we want out of this is a Jcobian or a gradient, that we could feed gradient descent to help to find the 
#minimum of the cost.

# But the kinda thing that we want is soething like this, where we have a vector, and each entry in the vector
#is how the cost changes with respect to some weight 1, some weight 2 up until weight n.

# And in our example we have 5 weights.

#   X1
#   X2 \
#   X3  --- N
#   X4 /
#   X5

# So our weights vector would generally have 5 weights, but for now, well just say up until weight n, after weight
#1 and weight 2.

#   __ __
#  | 2C  |
#  |---- |
#  | 2W1 |
#  |     |
#  | 2C  |
#  |---- |
#  | 2W2 |
#  |     |
#  | 2C  |
#  |---- |
#  | 2Wn |
#  |__ __|

# So we want this...

#   __ __
#  | 2C  |
#  |---- |
#  | 2W1 |
#  |     |
#  | 2C  |
#  |---- |
#  | 2W2 |
#  |     |
#  | 2C  |
#  |---- |
#  | 2Wn |
#  |__ __|

# To output this...

#                 _____
#                |
#     1          |    [0]T                         if WtX + b <= 0 
#   -----        |
#     m          |    m
#                |  ______
#                | \         (WtX + b - y) [X]T    if WtX + b >= 0
#                | /______
#                |    i=1 
#                |_____

# And now that we've taken the derivative, we just want to double check that that is what we're getting.

# So just to understand a little better, and just looking at this, (WtX + b - y), this is an error term, and
#(WtX + b) is our prediction, and remember that (WtX + b) is equal to aL, so basically aL is our prediction.

# So aL minus the corresponding real answer for the example is the error, so we'll just use this as a representation
#ei.

# So let's rewrite this to relfect the changes.

# First, in the case where the weight transposed times x plus b is less than or equal to 0 (WtX + b <= 0), remember we
#will have a vector of zeros, or zero transposed [0]T.

#   _ _
#  | 0 |
#  | 0 |
#  | 0 |
#  |_ _|

# So we got that, that's in the zeros case, if WtX + b <= 0, then our gradient with respect to its weights is zero
#because everything is zero.

# Now we can summize over the non zero case.

# So substituting this, WtX + b - y, for the error term we can now rewrite our new summation equation.

# We will have the error of i times x transposed.

# Remember that this, (WtX + b - y), is equal to our error term, that's why it can be substituted out.

#               m
#    1       ______
#  -----     \         ei [X]T    if WtX + b >= 0
#    m       /______
#              i=1 

# We can think of error term as how incorrect the answer is.

# So looking at the X transposed [X]T, and looking at the sum (i=1), let's say that we only have one example.

# Let's say our one example is going to be m = 1.

# With that, we can get rid of the summation, which will leave us with this.

#   1
# ------    ei [X]T
#   m

# And since we know that m = 1, we also say that   1          1
#                                                -----  is  -----  
#                                                  m          1

# This ccan be simplified to just 1.

# So what we are left with is,  ei [X]T

# In the case of our one example, m = 1, we just have our error term times X transposed.

# With our five inputs in error, this is what it will look like


#     __ __
#    |  X1 |
#    |  X2 |
# ei |  X3 |
#    |  X4 |
#    |  X5 |
#    |__ __|

# So when we multiple our one example, what we are going to do is multiple our error ei, by each input.

# Note: The error will remain the same for every training example.

# The i index is the training example.

# Remember we only have one training example m = 1.

# After multiplying the error term by every input, this will be our new vector.

#   __ __
#  | eX1 |
#  | eX2 |
#  | eX3 |
#  | eX4 |
#  | eX5 |
#  |__ __|

# So this vector here is how our cost changes with respect to the weights.  2C
#                                                                         ------
#                                                                           2W


#           __ __
#          | eX1 |
#   2C     | eX2 |
# ----- =  | eX3 |
#   2W     | eX4 |
#          | eX5 |
#          |__ __|

# And the reason why we have this vector, is because the whole cost function is just one big vector to scalar 
#problem.

# Remember in the beginning when we specified that we get gradients and multi variable functions that take vectors to 
#scalars.

# That's exactly what happened here.

# We put in our five X's as inputs, we put it through a neural network, and it spit out a single cost, which is the
#MSE (Mean Squared Error).

# The MSE (Mean Squared Error) is a mean of all the errors in our vector.

# So here is how c changes with respect to the weight with every input.


#           __ __
#          | eX1 | ---> How c changes with resoect to weight 1
#   2C     | eX2 | ---> How c changes with resoect to weight 2
# ----- =  | eX3 | ---> How c changes with resoect to weight 3
#   2W     | eX4 | ---> How c changes with resoect to weight 4
#          |_eX5_| ---> How c changes with resoect to weight 5

# BAsically, the dC/dW vector represent the ratios of change in C when changing the weights by some amount.

# Note: In larger networks, this dC/dW will become a matrix of derivatives the same size as the weights matrix W.

# So we have acheived our goal of knowing what happens when we change our weight, how does our cost function change.

# This is the answer

#           __ __
#          | eX1 | ---> How c changes with resoect to weight 1
#   2C     | eX2 | ---> How c changes with resoect to weight 2
# ----- =  | eX3 | ---> How c changes with resoect to weight 3
#   2W     | eX4 | ---> How c changes with resoect to weight 4
#          |_eX5_| ---> How c changes with resoect to weight 5

# Now what would happen if we had multiple training examples?

# We would basically extending vectors.

# Our [X]T will stay the same each time, but the error index (ei) is going to change, because each training example is 
#going to have a different error, depending on the answer or the inputs, all sorts of things, so it's going to have a 
#different error each time.

# But we want one vector.

# Let's say we have 3 training examples.

# We will multiple the error term for each training example.

# Then we will add those vectors together.

# This process will look something like this.

#   __  __     __  __     __  __
#  | e1X1 |   | e2X1 |   | e3X1 |
#  | e1X2 |   | e2X2 |   | e3X2 |
#  | e1X3 | + | e2X3 | + | e3X3 |
#  | e1X4 |   | e2X4 |   | e3X4 |
#  | e1X5 |   | e2X5 |   | e3X5 |
#  |__  __|   |__  __|   |__  __|

# Note: e1, e2, and e3 specify the three different training examples.

# Notice how the [X]T stayed the same on every training example.

# To simplify things, we can just make this one long vector.

#   _____________   ____________
#  | e1X1   +   e2X1   +   e3X1 |
#  | e1X2   +   e2X2   +   e3X2 |
#  | e1X3   +   e2X3   +   e3X3 |
#  | e1X4   +   e2X4   +   e3X4 |
#  | e1X5   +   e2X5   +   e3X5 |
#  |_____________   ____________|

# Now for the final step, we don't want to forget the 1
#                                                    ---
#                                                     m

# This will end up looking like this.

#   1
# ----- = e1X1   +   e2X1   +   e3X1
#   m

# Will do this for every row.

# For each of the rows in our long vector, we are going to take the average.

# So if there are 3 training examples we are going to divide the row by 3, and that will end up looking something
#like this.

#   e1X1   +   e2X1   +   e3X1
# ------------------------------
#              3

# And we will do this for every row in our long vector.

# So after averaging all of the different training examples, the errors times the X components, what we'll get is 
#just a general number that we can no longer really associate with [X]T, and with any of the values of X, because
#now it's just an average.

# Let's call this g1, g2, g3 g4, and g5, and we can represent it like this.

#   __ __
#  |  g1 |
#  |  g2 |
#  |  g3 |
#  |  g4 |
#  |__g5_|

# This gives a general idea of per these weights,over m training examples, what's the general when we tweak the Weight
#1 by something, what is the ratio of the corresponding change in the cost function, that is what g1 is.

# Basically, we can think of each g as an approximate derivative of the cost with respect to all Weights averaged
#over all training examples.

# So it's acutally quite beautiful what we get from this whole process.

# We get this long vector, that's as long as how many weights are going into our node, into how adjusting each one of 
#these weights effcts the cost that comes out.



# In This Section We Will Differentiating The Bias

#   Differentiating The Bias

# Similar to what we did with the weight, we'll do the same exact thing with the Bias.

# We'll find the derivstive of the cost with respect to the Bias now.

#   2C        1       ___
# ------   ------    \      (y - aL)squared
#   2b        2m     /___
#                     i=1

# So we're going ot do a similar thing where we segment off the y - aL to the intermediate function V.

# And using our same chain rule idea to find the derivative of C with respect to b, we need to chain it, which will
#give us something like this.

#   2C       2C     2V     2aL
# ------ = ------ ------ ------
#   2b       2V     2aL    2b

# First we'll see how v changes with respect to aL, which will give us a negative 1.

# Then we multiple that by how aL changes with respect to b.

# Earlier we established that this is how aL changes with respect to b.

#            _______________
#           |       
#           |   0 x 1   if WtX + b <= 0
#  2a       |
# ----  =   |        
#  2b       |   1  if WtX + b >= 0       
#           |________________


# So this is what we have now.


#                            _________________
#   2V     2aL              |
# ------ ------ = -1 times  | 0 x 1   if WtX + b <= 0
#   2aL    2b               |
#                           | 1  if WtX + b >= 0
#                           |_________________


# So -1 times 0 is, and -1 times 1 is -1.

# Now we will have this.


#                   _________________
#   2V     2aL     |
# ------ ------ =  |  0 x 1   if WtX + b <= 0
#   2aL    2b      |
#                  | -1  if WtX + b >= 0
#                  |_________________

# Now we can simplify this by canceling out the aL's to the change in V with respect to b


#            _______________
#           |       
#           |  0 x 1   if WtX + b <= 0
#  2V       |
# ----  =   |        
#  2b       | -1  if WtX + b >= 0       
#           |________________


# Now that we've covered the last two derivatives, we can now see how c changes when we change V.

#   2C       2C    
# ------ = ------ 
#   2b       2V  


# Now let's take the derivative of the cost with respect to b

#         _________________
#        |            m
#    2   |   1      ______
# ------ | -----    \        (Vsquared)
#   2b   |  2m      /______
#        |            i=1
#        |_________________


# The first step we can take is to say the derivative of the sum is the same as the sum of the derivative.

# That will give us a new equation

#                     
#              m      
#     1       ___      2
#   -----    \      ------  (Vsquared)
#    2m      /___     2b
#             i=1     


# Now we can use the chain rule to split up the Vsquared up into two functions.

# Soto find the derivative of Vsquared we can find the derivative of Vsquared with respect to V, and then the
#derivative of V with respect to b.

#    2Vsquared        2V
# --------------    ------
#      2V             2b


# This is the new equation we will have.

#                    _____________ 
#              m    |
#     1       ___   |   2Vsquared      2V
#   -----    \      | ------------   ------
#     2m     /___   |     2V           2b
#             i=1   |______________


# We can simplify this before we move forward.


#                    
#              m    
#     1       ___         2V
#   -----    \       2V ------
#    2m      /___         2b
#             i=1  


# We can break it down even further. We can remove the 2 from the 2V and the 2m


#                    
#              m    
#     1       ___         2V
#   -----    \        V ------
#     m      /___         2b
#             i=1  


# As we can see, we are left with a pure average of 1 over m, and V times the derivative of how V changes with 
#respect to b.


#              m    
#     1       ___         2V
#   -----    \        V ------
#     m      /___         2b
#             i=1 

# And we already know how V changes with respect to b because we already worked that out. (see example below)


#            _______________
#           |       
#           |  0 x 1   if WtX + b <= 0
#  2V       |
# ----  =   |        
#  2b       | -1  if WtX + b >= 0       
#           |________________


# So what we can do now is substitute them.


#              m         __________
#     1       ___       | 0 x 1   if WtX + b <= 0
#   -----    \        V |
#     m      /___       | -1  if WtX + b >= 0
#             i=1       |__________

# Now we can do V times 0, which is 0, and V times -1, which is -V.

# Now we have a new equation


#              m       __________
#     1       ___     |  0 
#   -----    \        |
#     m      /___     | -V  
#             i=1     |__________


# And now we have to substitute -V.

# And if we remember our V was equal to ( y - aL )

# So after the substitution of V, this is our new equation.


#              m       __________
#     1       ___     |  0 
#   -----    \        |
#     m      /___     | -( y - aL )  
#             i=1     |__________


# Now we can expand this further.

# Remember that aL is just our activation, so we can substitute it out.

# Y would be minusing the max of 0, Wtx + b

#              m       __________
#     1       ___     |  0 
#   -----    \        |
#     m      /___     | -( y - max(0, WtX + b) )  
#             i=1     |__________


# Remember that we learned that having the max of 0 is redundant since the piecewise functions will provide those
#answers.

# So we can simplify our code even further removing the redundencies.


#              m       __________
#     1       ___     |  0 
#   -----    \        |
#     m      /___     | -( y - WtX + b )
#             i=1     |__________

# Now we can distribute the negative, which will update our equation even further.


#              m       __________
#     1       ___     |  0 
#   -----    \        |
#     m      /___     |  ( -y + WtX + b )
#             i=1     |__________

# The last thing we can do is rearrange -y function to be more visually appealing.


#              m       __________
#     1       ___     |  0                 if WtX + b <= 0
#   -----    \        |
#     m      /___     |  ( WtX + b - y )   if WtX + b >= 0
#             i=1     |__________

# Just like last time, we can move the summation to the inside of the piecewise function.

# We can also move the mean inside the piecewise function.

# Let's go over what that looks like.


#          _____
#         |
#         |   0                           if WtX + b <= 0 
#         |
#         |          m
#         |   1    ______
#         | -----  \         (WtX + b - y)    if WtX + b >= 0
#         |   m    /______
#         |          i=1 
#         |_____


# So we wnd up with just this as our scalar. WtX + b - y

# And this makes sense because in our simple neural network of just one neuron, we only ave one bias to tweak,
#so that is our scalar.

# In a larger network where we have multiple layers, we would have a bias for each layer.




# Here we will be looking at the gradient descent intuition

#   Gradient Descent Intuition

# To start, let's go back to the single training example for a second so we don't have to worry about the averages
#just yet.

# The steps we are about to go over are applicable to when we are doing averages, but it's a little bit easier 
#to see it being written out here.

# This is a 5 dimensional vector.

#   __  __    
#  | e1X1 |   
#  | e1X2 |   
#  | e1X3 | 
#  | e1X4 |   
#  | e1X5 |   
#  |__  __|

# Let's think of this graphically.

# Let's take a smaller vector that we can graph. Something 2 dimensional.

#   __  __
#  | e1X1 |
#  | e1X2 |
#  |__  __|

# So this is a network with two weights attached to it.

# We can display this using a 3 dimensional graph.

#                   |
#                   |
#                   |
#                   |
#                   |
#                   |
#                   |
#                   |
#                   |
#                  / \
#                 /   \
#                /     \
#               /       \
#              /         \

# So we're going t be graphing e1X1 and e1X2, and see how the cost changes with respect to the weight

#      __  __
# 2C  | e1X1 |
#---- | e1X2 |
# 2W  |__  __|

# With this, theoretically, if we've been familiar with multi-variable calculus or gradients, we've probably heard
#that "The gradient of a function points in the direction of the steepest descent".

# And that gives us an intuition as to why this makes sense.



#                   |C
#                   |
#                   |
#                   |
#                   |
#                   |
#                   |
#                   |
#                   |
#                  / \
#                 /   \
#            W1  /     \  W2
#               /       \
#              /         \

# Here, W1 is the value we give to our first weight, W2 is the value we give to our second weight, and C is the 
#cost.

# So when we're trying to minimize the cost, we're trying to get to the bottom of this, but we're not going to focus
#on that now.

# The cost is represented but a sort of 3D shape (that we'll attempt to draw here).


#                   |C
#                   |
#             ------|-------
#             |     |      |
#             |     |      |
#             |     |      |
#             |     |      |
#             |     |      |
#             |     |      |
#             |    / \     |
#             |___/___\____|   
#            W1  /     \  W2
#               /       \
#              /         \

# We have to imagine that the representation of C is sort of like a curved blanket.

# There is an intuitive way we can look at this vector and why it points in the direction of steepest descent.

# First we will keep in mind that we want to get the gradient of the cost function with respect to the weights.

# Now we can start to get to why this points to the direction of the steepest descent and what tis means for us
#trying tofind the lowest cost.

# Something that's popualr in physics is viewing vectors as linear arrows through space.

# ------------>

# Where as here for example the X and Y in the normal cartiegian plane, the direction of a vector can be described by
#its X coordinate, just like a point on a cartiegian plane, with its Y coordinate.

# ----------->
# \        /
#  \      / Y
#   \    /
#  X \  /
#     \/

# So given that, we can travel from the origin (the point of X) to this place (the point of Y).

# So we can do the same exact thing on our 2D plane that is created by our W1 and W2 on our graph.

#                   |C
#                   |
#             ------|-------
#             |     |      |
#             |     |      |
#             |     |      |
#             |     |      |
#             |     |      |
#             |     |      |
#             |    /|\     |
#             |___/_|_\____|   
#            W1  /  |  \  W2
#               /___|___\
#              /  |   |  \
# Conponent of W1_|   |_ Conponent of W2

# Here we can see where the graphical conponents of W1 and W2 are connected.

# The conponenets that we get from this arrow are sourced from this vector

#      __  __
#     | e1X1 | W1 
#     | e1X2 | W2 
#     |__  __|

# So the amount given in the W1 direction is given by this (e1X1)

# And the amount given in the W2 direction is givenby this (e1X2)

# One more thing we can mention to keep in mind is with line vectors.

# As long as they're in the same direction, and the same length, we can move them anywhere in the plane.

# So what happens when we think of this 

#      __  __
#     | e1X1 |
#     | e1X2 |
#     |__  __|

#as a line in this plane.

#                   |C
#                   |
#             ------|-------
#             |     |      |
#             |     |      |
#             |     |      |
#             |     |      |
#             |     |      |
#             |     |      |
#             |    / \     |
#             |___/___\____|   
#            W1  /     \  W2
#               /       \
#              /         \

# Sincce there are only 2 elements in the vector, we can only down in the 2D portion of the graph (W1 and W2 region).

# If we wanted to go up in the third portion (the cost region), we would need a third element.

# Our idea is that the arrow on this plane points to the direction where the cost acescends the most steeply.

# So let's say, that in this direction it ascends pretty steeply. 

# So say that in this direction, its slope is steepest, for eaxample. (See example below)


#                    |C
#                    |
#             -------|-------
#             |      |      |
#             |      |      |
#             |      |      |
#             |      |      |
#             |      |      |
#             |      |      |
#             |     / \     |
#             |    /   \    |
#             |   /     \   |
#             |  / ----->\  |
#             |_/_________\_|   
#           W1 /           \  W2
#             /             \
#            /               \

# But why is this direction the steepest point?

# We can get an intuitive idea just by looking at our vector.

# One of the most important things we can notice, one of the most important observations we can get from deriving
#all the steps we took to get here, is that the cost, and how much tweaking some weight affects the overall cost,
#is extremely dependent on the input associated with that weight.

# We see that these E's will remain the same. (see example below)

#      __  __
#     | e1X1 | W1 
#     | e1X2 | W2 
#     |__  __|

# These E's only vary from training example to training example.

# All the E's in a particular training example will be some constant number.

# For example, if we had multiple E's, each example would have its own constant, (e1, e2, etc etc...)

# The things that will change within each example are the values of the inputs, (in our case the X's).

# The value of the inputs is what decides the derivative, or how much changing the weight affects the cost.

# So if the input is very large, even a small change in the weight will cause a very large change in the overall 
#cost.

# That can be a bad thing.

# Changing the input even a little can result in a drastic change to the cost.

# As opposed to a smaller X, let's say X2 for example in our case, any multiple of error against this will result
#in a much smaller change.

# We can almost think of it as changing W2 is less significant as changing W1, because the change is much more
#drastic, because even a small change to W1 will cause large decrease in cost.

# And this is really the heart of optimization, with gradient descent.

# We're trying to figure out how we can get the most bang for our buck by seeing what derivatives we can change
#the easiest with even small tweaks to the W.

# Often with training a neural network, the ones with very small inputs will just kinda be dead.

# The nodes won't really learn much. They might go up by a very small percentage, but not enough to learn a lot.

# So back tomour graph.

#                    |C
#                    |
#             -------|-------
#             |      |      |
#             |      |      |
#             |      |      |
#             |      |      |
#             |      |      |
#             |      |      |
#             |     / \     |
#             |    /   \    |
#             |   /     \   |
#             |  / ----->\  |
#             |_/_________\_|   
#           W1 /           \  W2
#             /             \
#            /               \

# If we imagine our arrow in space on the plane, when X1 is big, that means that the error (e) is magnified.

# So if X1 is big, and even if the error (e) is small, the error (e) can be quite large, which can negatively
#affect the cost quite badly.

# So when we're creating the vector, what it's going to do is it's going to point in the direction of W1 if X1
#is very large.

# Basically, the arrow points in the direction of the weights that have the most impact, and therefore have the 
#biggest cost, because their error (e) is magnified by the X1, for example.

# In contrast, when we multiple the error (e) by X2, which is a smaller value, it has a much smaller affect 
#on the overall cost, so the arrow wouldn't point in that direction as much, because pointed in that direction
#wouldn't be pointed towards the steepest part of the cost.

# Pointing towards the part with the steepest cost would be pointed towards the magnified error (e).

# Why do we want to find the direction of the highest cost when we want to be headed in the direction of the lowest
# #cost?

# Because using the direction of the highest cost will enable us to find the direction of the lowest cost.

# And to do this is simple.

# Taking the negative of the gradient tells us how to most quickly decrease the cost.



#   In this section we will be talking about the Gradient Descent Algorithm and Stochastic Gradient Descent.

#   Gradient Descent 

# SGO

# Now that we've calculated all these gradients so we have this long vector with respect to all our weights and
#biases.

# So we have our cost in respect to our first weight, cost with respect to our second weight, then the cost with
# #respect to our bias, all the way up until some finally weight or finally bias, it really doesn't matter.

#   __  __
#  |  2C
#  | ----
#  |  2W1
#  |
#  |  2C
#  | ----
#  |  2W2
#  |
#  |  2C
#  | ----
#  |  2b1
#  |
#  |  2C
#  | ----
#  |  2Wn
#  |__  __


# So, this is the gradient that we calculate by doing that entire process of Jacobians and Chain Rules that we 
#covered.

# And we combine it all in this vector, so all of the information we need is in here.


#           __  __
#          |  2C
#          | ----
#          |  2W1
#          |
#          |  2C
#          | ----
#          |  2W2
# f(x,y) = |
#          |  2C
#          | ----
#          |  2b1
#          |
#          |  2C
#          | ----
#          |  2Wn
#          |__  __

# Now let's see how we can use this gradient, and equipped with that graphical understanding of the gradient points
#and direction of steepest descent of our cost function.

# It points in the direction of how we can get C the highest as quick as possible. 

# It makes sense that our formula for gradient descent looks like this, where each function is some weight or some 
#bias.


#           __  __
#          |  2C  | 
#          | ---- | W1 or B1
#          |  2W1 |
#          |      |
#          |  2C  | 
#          | ---- | W2 or B2
#          |  2W2 |
# f(x,y) = |      |
#          |  2C  |
#          | ---- | W3 or B3
#          |  2b  | 
#          |      |
#          |  2C  |
#          | ---- | Wn or Bn
#          |  2Wn |
#          |__  __|


# So if we have all our weights and biases in one vector, we can unroll.

# Something we do if we want to quickly talk about how the code works in a neural network is when we're about 
#to do gradient descent, we unroll. 

# We know we have all of those weights matrices, what we do is basically turn all of those weight matrices into
#one super long combined long vector, that's the exact same length as the gradient of the function.

# So we would have all our weights and biases in one vector, then we would have our gradient, and they would
#be the exact same length, so we could peform element-wise operations on them (see example below)

#   __  __      __  __
#  |  W1  |    | grad |
#  |  W2  |    | grad |
#  |  W3  |    | grad |
#  |  B1  |    | grad |
#  |  B2  |    | grad |
#  |  B3  |    | grad | 
#  |__  __|    |__  __|

# The gradient descent algorithm uses this to its advantage.

# We'll go over an example of how below.

# Let's use the capital theta to represent all the weights and biases in our entire network.

#   (-)

# Gradient descent is an iterative algorithm, so we don't just do it once and find a solution.

# We keep doing the algorithm over and over until thecost gets lower and lower, until we reach the lowest cost.

# And why do we do this?

# Let's say we randomly assigned our thetas weights and biases in the beginning before training and our
#cost was really high.

# So what we're going to do is, take the same theta and we're going to update it by an amount.

#   (-) = (-)

# So we're going to negative alpha, then our gradient.

# This will be our new algorithm for gradient descent.

#                     ___
#   (-) = (-) -aplha  \ / w,b C

# We will iterate this over and over again.

# So this is what we're doing in that process.

# We have all of our weights and biases, (-).

# Then we update those weights and biases, (-) = (-).

#                                                                               ___
# We update by subracting some learning rate times the gradient, (-) = (-) -aplha  \ / w,b C

# And this makes sense, because we're basically taking the negative gradient (ignoring the learning rate),
#which is pointing in the direction of how we can decrease the cost as quickly as possible, and that's what we're
#looking for.

# We're looking for how we can push our weights and biases in the direction of the arrow that's pointing towards
#the lowest cost on the graph.

# So we're kinda shifting the arrow that may be pointing in a different direction towards the right direction
#which has the lower cost.

# Note: By "arrow" we mean thinking of the total vector of parameters occupying the same geomeric space as our
#gradient arrow

# By shifting it by way of the lower cost, we're just subtracting (whcich is our element-wise operation) Weight 
#by the derivative Cost with respect to W, multipled by some learning rate (alpha)

# And we would do the same process for all our weights and biases. W1, W2, B1, B2, etc etc...

#   _______   _____
#  |          2C   |
#  | W1 - (a)----  |
#  |          2W   |
#  |               |
#  |               |
#  |_______   _____|

# That's basically what we're doing here.

# We're subtracting the corresponding partial derivative from the weight, and this will slowly nudge us in the
#right direction and get us closer to our solution.

# So the learning rate plays a critical role in how quickly we make these edits to our weights and biases
#and how big those edits are.

# Finding the learning rate is a hyperparameter, because it is an algorithm that the engineer has to figure out
#themselves.

# In the classic sense that the gradient descent is first taught is that we would do gradient descent over our entire
#training batch.

# The larger these training batches are, the more computational expensive it will become for us to do gradient
#descent o our algorithm.

# So what the leading tactic now is, is batching up our training sets into multiple smaller batches.

# So let's say we have X which equals 1,000,000 training examples.

# We could batch this into batches of 120.

# So we divide 1,000,000 by 120, and the result would be how many batches we have, which would be 8,333 batches.

# This would look something like this.

# {X1}, {X2}, {Xn}

# Each one of those batches would represent 120 training examples.

# So what we're going to do is stochastic gradient descent on the first batch.

# And then we'll do the stochastic gradient descent on the second batch.

# We will continue this process for every batch we have.

# We have effectively cut down our 1,000,000 training examples down to 8,333 batches of 120 training examples.

# What this means is that instead of taking one step and doing gradient descent on 1,000,000 training examples,
#we can take 8,333 steps and do gradient descent for each step, which is computationally more cost effective.

# This Version of gradient descent is known as Mini Batching.

# Next we will try to explain how we can get from the X1, X2, X3, Xn single node system, to a system with many more
#nodes, and seeing how that works and how we can do similar things to what we did with one node with multiple nodes.



#       Finding Derivatives of an Entire Layer (and why it doesn't work the way we did with one neuron)

#   Derivatives in Larger Networks

# So now that we finished documenting the derivatives of the weights and the biases of a small neural network,
#let's see how this works with slightly larger neural networks.

# So let's take something like the example below where we have 3 inputs connected to 3 neurons.

# X1    O

# X2    O

# X3    O

# These are in return connected to 4 neurons.


              
#            O
# X1    O   
#            O
# X2    O   
#            O
# X3    O   
#            O


# And finally, these 4 nodes get fed into an output node, which gets computered as some sort of cost.


#            O
# X1    O   
#            O
# X2    O           O
#            O
# X3    O   
#            O


# So we would have 3 layers, which would be connected by 3 weights matices (Remember that our inputs and matrices
#are connected by weights).

# So how would we calculate each one of these derivatives of the cost with respect to the weights and biases.

# So theorectically if we're just using this kinda chain rule intuition, if we wanted to find the entire cost
#with respect to the first set of weights, we would do something like find how the cost changes when we change 
#the first set of weights.


#   2C
# ------
#   2W1


# Then we would see how the first layer of activations change when we change the first weight.


#   2C       2a1
# ------ = ------
#   2W1      2W1

# We would do this for every layer, (a1, a2, a3) and then we would see how the cost will change with respect to
#the last layer, (a3).


#   2C       2a1      2a2      2a3      2C
# ------ = ------ = ------ = ------ = ------
#   2W1      2W1      2a1      2a2      2a3


# Then, if we wanted to find the weights too, we would have to do the same process for each weight.


#   2C       2a2      2a3      2C      
# ------ = ------ = ------ = ------ 
#   2W2      2W2      2a2      2a3     

# Note: We start at a2 here and not a1 because of the atrarting position of the W2.