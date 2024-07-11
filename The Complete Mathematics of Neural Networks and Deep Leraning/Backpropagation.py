# In this section we will be going over Backpropagation and everything that comes along with it.

#       The error Of A Node

# Now that we're talking about backpropagation it's really important that we talk about the concept of the error of a 
#single node.

# So let's start be taking a look at a neural network.

#   O    O

#   O    O    O

#   O    O


# This neural network has 3 inputs.


# X1    O   O

# X2    O   O   O

# X3    O   O


# The error of a node is choosing some node, (the second node in the first layer (a1 layer)), choosing this node which
#can be represented as W transposed X plus B, (r(WtX + b)) (Note: r is for the relu activation).

# So this would be represented like this now, a1,2 = r(WtX + b)

# If we were to add some small change to that node, how would the overall cost change.

# There is one more specificity we need to make here.

# We're not actually talking about a.

# We're not actually talking about adding a small quantity to the entire value of a itself.

# But for mathematical reasons we'll get the same answer but just adding a small change in z.

# z = (WtX + b)

# So really we're talking about adding a small change in the z1,2 value of that node.

# So basically the activation of that node before it goes through the activation function.

# So the z1,2 of that node.

# So if we add some small quantity to that node, how will the overall cost function change?

# That's the idea of the error of a node.

# The error of a node can be described by this symbol & with l and j (&l,j), so some layer l with the jth neuron,
#is describe by the change in the cost function when we change the z of l and j


#            2C
#   &l,j = ------
#            2Zl,j


# In our situtaion, the error of that node & 1, 2, equals how the cost changes when we change the Z 1, 2 of that node.

# This is just some important information to keep in mind.

# One last question would be why this makes sense, and why we call it the error of a node.

# We can think about it like this.

# If the error derivative C with respect to Z l, j is close to zero, that means changing the value Z has little
#effect on the final cost.

# Alternatively, if it's quite a large value, changing the value of the node creates a large change in the cost.

# So kinda going back to the bang for our buck idea, if we wanted to decrease the cost, we would take the 
#negative of the derivative of the larger cost and that would actually reduce the cost by quite a bit.


#   The Four Equations of Backpropagation

# The reason why we're outlining these 4 central equations to backpropagation is that they basically make up the whole of backpropagation, and
#once we know these 4 steps backpropagation will become much more easier.


#   1st Equation

# So using the intuition of some nodes error(&), some nodes j of layer l (l, j),  = to how the cost changes when we change 
#this nodes Z(l, j) value.

#
#    l      2C
# &    = ------ l
#   j         2Z
#               j

# Let's see if we can find the error of the node that is in the last layer of its network.

# So it's getting directly compared with our real answer y.

# So for example let's say we have this newtwork.


#   O   O

#   O   O   O = C

#   O   O


# Notice that the last layer is our output layer and our final answer.

# This final activation (aL) which is acutally equal to the activation function of zL ( aL -r(zL) ).

# How does this overall cost, which is a kind of variation of the real answer minus aL squared. (y - aL)squared

# How does that cost change when we change Z?

# So what we're trying to find is how does that cost change when we change Z of the last layer.

# Even if there are multiple nodes in the last layer, we want to find how does the cost change when we change the Z in
#the last node of the last layer.

# So we're looking for the capital L, which is the last layer, and some j.

# And this can be equal to how our cost changes with some activation node in the final layer a(L, j) multiplied by how that
#activation node in the final layer changes when we change Z(L, j)

#  L     2C         2a(L, j)
# &  = ------     -----------
#  j     2a(L, j)   2z(L, j)
#                      |
#                      |
# So simple enough, this will just be the derivative of the activation function with respects to (L,j), which will
#then get multiplied by whatever the cost function is.


#   2C
# -----       .    r( z(L, j) )
#   2a(L, j)

# And whatever this ends up being is our (L, j)


#  L     2C       
# &  = ------     .  r( z(L, j) )
#  j     2a(L, j)

# This would be considered a conponent wise kind of equation, for this error of the last node, because we're 
#lookng at specific nodes j in the ending layer.

# But really for backpropagation we want to see more matrix based notation, so we'll be dealing with the entire layer.

# So now let's find the error for every node.

# So we'll have a vector of errors.

# So all we need is the layer

#      ___
#     |
#   L |
#  &  |
#     |
#     |
#     |___

# So we're looking for the error of each node in the layer.

# So now we're looking for this.

#  __________________________________
# |  L
# | &  = 
# |  
# |__________________________________

# So if we taking the derivative of the cost with respect to all of the nodes in the last layer aL, we're taking
#the partial derivative of the cost with respect to all the different aL nodes.

# So what we're doing is we're actually getting gradient of the cost with respect to a.

# Before we were looking for the gradient of the cost with respect to w and b when we were trying to derive the
#cost with respect to the weights and the biases.

# But now we're just doing it with respect to all the values in aL.

# So what we can do is take the derivative and display it notationally as a gradient, because that's exactly what 
#it is.

#  __________________________________
# |  L     ___
# | &  =   \ /(a)C
# |           
# |__________________________________

# And then we can multiple this by the same thing, but the only difference is we will just have the L.

#  __________________________________
# |  L     ___
# | &  =   \ /(a)C r( Z(L) )
# |           
# |__________________________________

# We can think of this as an elemental wise.

# We're bascically we're finding what derivative of whatever our activation is function is, and then we're plugging 
#each element from our Z vector which is going to be a vector of all the Z values of the nodes in our final 
#activation layer.

# And we're just going to be elemental wise applying the derivative of our activation to Z1, Z2, etc.


#      ___
#     |
#   L | r1 (Z1)
#  &  |
#     | r1 (Z2)
#     |
#     |___

# It's all going to be an element wise operation.

# An element wise multiplication between each item in the gradient and each item in whatever this vector (r (zL) )
#ends up being, which is going to be this (see example below.)

#      ___
#     |
#   L | r1 (Z1)
#  &  |
#     | r1 (Z2)
#     |
#     |___

# And this is going to be our vector of errors.

# This is the first equation in our matrix.



# Here we will be going over the second equation, which is the error of any node.

#           Find The Error Of Any Node

# So now that we know how to find the error of the nodes in the final layer, how can we use that to find the error
#of any node in our network.


#  L    ( L + 1 )T  L + 1         L
# &  = ((W      )  8      ) O r (Z )

# This formula tells us how if we have some layers of our neural network, and if we know the errors of the nodes in 
#layer L + 1, we can use these to derive the errors in layer L.

# So using this formula we can back propagate through a network and kinda cumalatively find the errors with respect
#to each of the nodes, and this formula tells us how we know them.

# So this formula tells us that if we have the weights matrix of L + 1, that's a weights matrix that connects L + 1
#and L (remember that the weights matrix syntax is that the superscript is the layer it's going into, so if it's 
#going into L + 1 that means it's coming from L).

# So the weights matrix that tells us what all the weights are, if we transpose the weights matrix (we flip it) and 
#multiple it by the vector of errors of L + 1, and multiple that element-wise.

# If we multiple that element-wise by the errors of the derivatives of activations, (the ones that have not been touched 
#by the activations yet) then we can get the error.

# So what will outcome is a vector of errors, which have the same amounts of entries as the amount of nodes in that
#layer, and each entry in the error of L vector is the error associated with each node.

# So explore two reasons why this works out.

# One is super intuitive, and we will discussing this one first.

# When we multiple the errors of a layer L + 1, we can think about it as when we multiple by the transpose of the
#weights matrix that connects the two layers we can almost think of it as going in reverse and backpropagating 
#the error to the layer before it.

# If we look at the dimensions it makes sense, and if we look at how the weights matrix sort of hints on the intuition,
#in the weights matrix, the columns are the (k).

#             K
#      ______   ______
#     |               |
#     |               |
#  W  |               |
#     |               |
#     |______   ______|

# Remember when we saw the (J, K, L), the K is the column it came from, and the J is the column it's going into.

# So if we transposed this and we now had this.

#             J
#      ______   ______
#     |               |
#     |               |
#  K  |               |
#     |               |
#     |______   ______|

# Now the columns is where it's going into.

# So when we're thinkinging about mutiplying this, we're talking about reversing the process of feed forward, so now
#we're backpropagating the errors through the network to the layer before it.

# And that's intuitively what we're trying to do, we're trying to find the errors of a layer by cumulating the errors
#of the layer after it and seeing how they effect the layer.

# And then we multiple the activation of Z(L), because remember how the error is defined as how the cost changes with 
#respect to Z. 

# And then if we're thinking chain rule wise we can't go staright to the aL, so we don't have to multiple by the zL.

# But something that makes a lot more sense is the chain rule intuition.

# This way we can fill the gaps of the chain rule in a way that makes sense.

# So let's start off with an example and see if by trying to find the error in our example, we get something that
#completely matches this formula (see formula below)

#  L    ( L + 1 )T  L + 1         L
# &  = ((W      )  8      ) O r (Z )


# So let's look at our example where we're going to take the same notation that we talked about in the beginning
#of this lecture, where we're going to represent the entire layers of a neural network in a kind of horizontal 
#way.

# So let's say we have some weights matrix 1 and some vector a0 of input (that could be said to x, but we'll keep it
#at a0), so this is the first layer in the entire neural network, plus the bias, which is also a vector.

# So this is a matrix (W1), this is a vector (a0), and this is a vector (b1), And this is epresenting an entire
#layer.

# r(W1 a0 + b1)

# And now we can go to the second layer and see how we're using a new weights matrix (W2), we're multiplying the
#the output of this layer of nodes ( r(W1 a0 + b1) ), which is a1, and then we're adding a new bias vector (b2).

# r(W1 a0 + b1) + r(W2 a1 + b2)

# Now we have two layers represented in this way.

# So how can we try and find the error of the nodes in layer 1?

# That is the goal. Seeing if we can find the error in all the nodes in layer 1.

# First, Let's keep in mind that it's the Z1 we're looking for.

# The change in the cost with respect to the change in Z1.

#   2C
# ------
#   2Z1

# So using this definition of the error, how can we dervive from the chain rule what's missing.

# So what we're trying to find is some change in the cost with respect to Z1.

#      Z1              Z2
# r(W1 a0 + b1) + r(W2 a1 + b2)

# So given the error of Z2, assuming that we're given the error of the layer in front of Z1, because we're 
#backpropagating, so say we some how already calculated the errors of Z2, so we have the change in C with respect
#to the change in Z2, somehow we have that.

# How can we get from Z2, to Z1?

# Remember how when we're doing the chain rule where we have some f and how that changes with respect x, and how
#we're able to split that up into how f changes with respect to u and how u changes with respect to x, and then
#kind of cancel those out, we can kinda do the same thing now.

#   2f       2f      2u
# ------ = ------  ------
#   2x       2u      2x


# So now we're seeing that we have the cost and the numerator, how can we get Z1 somewhere in the denominator so
#that they all cross out and we end up with this

#   2C
# ------
#   2Z1


# That's kinda what we're doing, some detective work on how we can fill these gaps, connecting how the cost changes
#in Z2 to how the cost changes in Z1.

# So we're going to think about what changes in Z2 effects some sort of change so that we can go backwards in our 
#network. 

# The best kind of link towards Z1, the most obvious link would be matching a numerator to our Z2 denominator to see 
#how Z2 changes when we change a1.

# a1 connects us back to Z1.

# Basically we're trying make the links in our chain so we can get to Z1.

# So what we're trying to do is see how the cost changes with respect to Z2, so that's the error we're given.


#   2C       2Z2     2C
# ------ = ------  -------
#   2Z1      2a1     2Z2
#                     |                                                  L
# So this ----------(2Z2), is equivalent to the error term of layer 2 ( &  ).

# And then we multiply with how Z changes with respect to a, kind of chaining our way back.

# So now we can see how (a1) changes with respect to (Z1).

# We see that if we do some cancellations, and we're left with the change in C with respect to the change in Z1.


#   2C       2a1     2Z2      2C
# ------ = ------  -------  ------
#   2Z1      2Z1     2a1      2Z2
#    |                         |
#    |                  1      |                   2
# This is the error in &     This is the error in &

# So how does Z2 change with respect to a1?

# So how does this (W2 a1 + b2), change with respect to a1.

# We can eliminate b2 from the picture first which becomes 0.

# Next we'll work on the W2 matrix, and the a1 vector, which will give us a matrix vector product.

# Since we have a matrix vector product, and we're taking the derivative with respect to the vector a1, our 
#derivative is going to be W2T. 

# And then the a1 over Z1 is just how this entire function a1 (r(W1 a0 + b1)), changes when we change the Z1, so
#that's obviously a chain rule to do with the activation function.

# So how a1 changes with respect to Z1 is just going to be the derivative of our activation function, whatever
#that is, with our Z1 inside there ( r(Z1) ).

#                                                                             2
# So that's what we're doing. We're multiplying these 3 elements ( r(Z1) W2T & )

# So what we're doing, we're trying to find the change in the cost with respect to Z1, which is the error in layer
#one, which is equivalent to the change in a1 with respect to the change with Z1, the change in Z2 with respect
#to a1, and the change in the cost with respect to Z2.

#                                                                                              2
# Now we find that these are entirely equivalent to the multiplication of these 3 ( r(Z1) W2T & )

#   2C       2a1     2Z2      2C
# ------ = ------  -------  ------
#   2Z1      2Z1     2a1      2Z2
#             |       |        |
#             |       |        |2
#           r(Z1)    W2T       &


# Where have we seen this before?

# Let's take a second to check what these mean.

# The order of these can be flipped around.

# Also, this is an element-wise product r(Z1) times W2T.

#                               2
# These produce a vector W2T - & , and they are in the correct order according to our original equation, where 
#we have our weights matrix times our errors.

# It doesn't matter if we do r(Z1) before or after because it is element-wise product, but we want to do these
#W2T - &(2) first.

# So where have we seen this before?

# We're trying to figure out the error in layer 1.

# So our L equals 1 (L=1).

# So we're multiplying the weights matrix of layer 2, which is L + 1, and we're transposing it.

#                                                                            L+1
# Then we're multiplying by the errors in the next layer which we also have &    , then we do an element-wise of 
#product of things in zL which is Z1.

# So we find that we automatically find this formula whenever we try to use the chain rule to find the errors
#of our layer with respect to a layer that comes later.

# That's it for our second equation.

# That's an intuitive way of how we can backpropagte through a network and calculate all the errors just kind of
#cumulatively.

# This kinda gives us an idea of why we use backpropagation in the first place.


# Next we will look at our third equation.

# Now we're at the part of the equation where we actually solve for the weights and the biases, or the cost with
#respect to any weights and biases

# These are the things that we are actually going to put into our gradient descent algorithmn to help solve
#or improve our neural network.

# So this is what we're looking for.

#   2C
# ------ =   L
#     L     &      
#   2b


# Often these vector of biases for a single layer, so a single layer in a network where we have four neurons.

# So if we have four neurons, the vector of biases for this layer will be four long. One for each node.

# Often this is the same for each layer so it would be something like 4, 4, 4, and 4.

#   __ __       O
#  |  4  |      
#  |     |      O
#  |  4  |      
#  |     |      O
#  |  4  |      
#  |     |      O
#  |__4__|

# But from the formula above, we see that this bias vector is equal to the error vector of that layer.

# So if the error for each node is the is going to be 4, then the biases are going to be 4.

# Or vice versa.

# So they're always going to be equal to each other.

# So let's see why this is.

# We're going to use a similar intuition as we did with the previous equation.

# We're going to take an example and see if we can find this format.

#   2C        L
# ------  =  &
#   2b(L)

# So let's take two layers of our network which we write in this style.
# r(W1 a0 + b1) => r(W2 a1 + b2)

# Now we have our layers.

# let's say we want to find the bias of layer 1.

# The bias associated with a0(layer 1).

# How do we go about doing this?

# Using the two equations previously, we were able to find all the errors with respect to any layer.

# So again, assuming that we have the errors for every layer, and we are trying to find the bias for a0(layer 1),
#we already have the error associated with that layer in our equation, which is &(L)

# So we already have this value[ L]
#                              [& ]

# Whatever that vector is.

# And this vector is completely equal to how the C changes with respect to changes to Z1.

# 
#    L     2C
#   &  = ------
#          2Z1
#

# Knowing this we can just do some chain rule work.

# So if we are given the error term, and we want to find how the cost changes with respect to the bias of 1,

#   2C       2C
# ------ = ------
#   2b1      2z1
 

#what could we multiple this with?

# How does b1 affect z1?

# That's exactly the question we are going to be asking.

#   2C       2z1     2C
# ------ = ------ ------
#   2b1      2b1    2z1


# And after cross eliminating, we're done.

# That's our very short chain.

# And all we really needed to solve it was this term, how z1 changes with respect to changes to b1.

# That was equation 3.


# Here we will be going over equation 4.

# Finding the derivative of a cost with respect to any bias.

# Looking at this equation, there is one thing that we might notice.

#   2C            L-1         L
# ------    =   a(K   )    &(1  )
#       L
#   2W(J,K)


# This isn't a vector equation, but we're getting a scalar.

# So we're inputting a scalar for the error of node J layer L, and then we're multiplying that by the K activation
#node in layer L - 1, and then that gives us the derivative of the cost with respect to the weight (J, K).

# So let's take an example network.


#  L-1  L
#   O
#       O
#   O
#       O
#   O
#      _O
#   O_/ Example 1

# So this represents two layerss in our neural network.

# Let's say that we're trying to find how changing  the weight between the last nodes in layer L and layer L - 1
#affects the cost further down the road. (see example 1 above for which weight we are refering to)

# And let's say that our weight represented by just W. We're trying to find how the cost changes with respect to 
#this specific W.

#   2C
# ------
#   2W

# So how do we index weights?

# We want to find some W(J,K,L).

# The formula above gives us the derivative of the cost when changing a single weight.

# So this is a scalar derivative. It tells us how the cost changes when we change a single weight in layer L 
#connecting to k node of of L - 1, to the J node of L.

# So we have weight(W) (J,K) in layer (L). 

# We're trying to find how this changes the cost.

# We can kinda think of this slightly differently.

# What do we have already?

# We have the error term of how the cost changes when we change the error in layer L on node J.

# So we have the derivative of how Z on the last node of layer L and how it affects the change in cost.

# So we have how the cost changes when we change Z(L,J)

# So if we're thinking about the chain rule and we're trying to find how the cost changes when we change 
#W(J,K,L), we already have how the cost ties in later down the road, in the form of our error term, Error of L and J
#&(L,J).

# So we have how Z of L of J changes. Z(L,J)

# But we want to find how Z(L,J) changes when we change the weight(W), which is going to be weight (J,K,L)


#   2C            2Z(L,J)         2C
# ------    =   ------          ------
#   2W(J,K,L)     2W(J,K,L)       2Z(L,J)

# Now we can do cross cancellation.


# Now we will be looking at Vectorization

# This is still apart of Equation 4

# Here we will be finding the derivative of a cost with respect to any bias, VECTORIZED

# One last thing to point out before we move forward is that in the equation 4 scenario, we used a scalar, as
#opposed to the first 3 equations where we worked on vectors.

# With that being said, we will be trying to find some formula that is equivalent to our original equation,
#but is vectrorized. (See line 609 for our original equation)

# So we're dealing with vectors and looking for the derivative of the cost with respect to all weights in 
#a layer.

# But it's a little different with weights.

# Remember how we represented weights with a weights matrix, where W(subcript L) describes some matrix of
#all the weights.


#       ______   ______
#      |               |
# W  = |               |
#  l   |               |
#      |               |
#      |               |
#      |______   ______|


# Let's have an example of how we can find this formula.

# So let's say we have 3 nodes in one layer, and then 3 nodes in the next layer.

# And then we have all of our connections between us.


#  L-1  L 
#   O   O
#
#   O   O
#
#   O   O
#

# The amount of connections and weights is gonna be 9.

# That's because it will be a representation of 3 nodes being connected to each of the 3 nodes in the next layer
#individually, which can be represented by the equation 3 X 3.

# The shape of our weights matrix is also going to 3 by 3.

# More importantly, the columns of the weights matrix is the node it's coming from.

# And then the row of the matrix is the node it's going into.

#              K
#       ______   ______
#      |               |
# W  = |               |
#  l   |               | J
#      |               |
#      |               |
#      |______   ______|
#            3 by 3

# So something at the K column and J row is indicating that it's coming from the K neuron in L - 1 and going into 
#the J neuron in layer L.

# So our derivative of the cost with respect to weights matrices should be something with the exact same dimensions
#as our 3 by 3.

# So right now this is what our weights matrix looks like with respect to all the weights in our two layers 
#of neurons.

#              K
#       ______   ______
#      |  W11 W12 W13  |
# W  = |               |
#  l   |  W21 W22 W23  | J
#      |               |
#      |  W31 W32 W33  |
#      |______   ______|
#            3 by 3


#                                                                    L
# Looking at the index of our original equation we remember that 2W(J,K) is how we read the value in the weights
#matrix.

# For example, W12 is saying that the weight is going into the J node(1) in layer L, coming from the K node(2) in 
#layer L - 1.

#  L-1  L 
#   O   O
#     / = W12
#   O   O
#
#   O   O
#

# That woud be the weight connecting the second node in L - 1 ( first node of line 779 ), to the first node of L
#(second node of line 777).

# So what we're looking for is some change in cost with respect to W(subscript L) which is going to be equivalent to 
#the change in the cost when we change W11.


#           _________    _________
#          |  2C      2C      2C  |
#   2C  =  |------  ------  ------|
# ------   | 2W11    2W12    2W13 |
#   W      |                      |
#    L     |  2C      2C      2C  |
#          |------  ------  ------|  
#          | 2W21    2W22    2W23 |
#          |                      |
#          |  2C      2C      2C  |
#          |------  ------  ------|
#          | 2W31    2W32    2W33 |
#          |______________________|

# So basically a matrix of derivatives which is the same size as our weights matrix which tells us how each of the
#the weights changes with respect to the cost.

# So that will kinda finish our picture of vectorizies the 4 equations that we went over.

# And in the final part we can look at how we can actually do the backpropagtaion algorithm with vectors, because
#when we're actually doing backpropagation in neural networks, we're not dealing with single numbers, we're dealing
#with vectors because it is a lot more efficient.

# So how can we find this matrix of derivatives of C with respect to some W(subscript L)?

# And how do we come with a formula that is like this but calculates all of the weights at the same time.

#   2C            L-1         L
# ------    =   a(K   )    &(1  )
#       L
#   2W(J,K)


# First let's see what our matrix should look like, then we can try to find some vector product, or something that get 
#us there.

# We'll be using this example here as our network. (Imaginge that all our connections are made and we have our weights)

#    1     2 
#   (k)   (j)
#
#   (k)   (j)
#
#   (k)   (j)
#

# We're trying to find how the cost changes with respect to the weights of layer 2 in our network which is signified
#by W2. Basically how the cost changes when we change all the weights in W2.

#   2C
# ------
#   2W2

# Because the matrix label is the label it's going to.

# So using this rule (see line 818), let's just hand compute for each of the nine weights what these are going to be 
#and put them in our matrix.

#              K
#       ______   ______
#      |  W11 W12 W13  |
# W  = |               |
#  l   |  W21 W22 W23  | J
#      |               |
#      |  W31 W32 W33  |
#      |______   ______|
#            3 by 3


# So for W11, using the scalar formula (see line 818), what will the derivative of the cost be with respect to W11.

# Whatever the answer is, it's going to be a product of two scalars, the activation of L - 1 ( a(L - 1) ), which is 
#always going to be 1, so we can just fill (a1) into our matrix for every position.

# That is because in all of these formulas we will be multiplying by the activation of L - 1.

# The activation number of the neuron in layer L - 1 might change, but we will multiple by the activation of L - 1.

# So back to the process, we'll start with W11, although it usually doesn't matter which weight we begin with, we'll
#start here.

# To find W11, we can start by plugging in the numders in the formula.

# J = 1 and K = 1.

# We know that the weight is coming from K, and going into J.

# Now we will multiply a(1,1) because L - 1 = 1(layer), and K = 1. That's why we have a(1,1).

# We will be multiplying that by the error in layer 2, J = 1. &(2,1). That's because the error is in the second
#layer, and that is the layer that the weight is going into, which is J, and J = 1.

# Following these  steps we can get the rest of the weights in our matrix.

# Note: Every row will have a different error number to represent which node we are working on. For example
#the first row was all error (2,1). 2 represents the layer, and 1 represents the node that we are working on.

# The second row will be error (2,2)

# And finally the third row will be error (2,3).

# The activations will keep the same pattern for every row, a(1,1), a(1,2),and a(1,3).


#           __________     __________
#          |  2   1 |  2   1 |  2   1  |
#   2C  =  | &   a  | &   a  | &   a   |
# ------   |  1   1 |  1   2 |  1   3  |
#  2W2     |        |        |         |
#          |  2   1 |  2   1 |  2   1  |
#          | &   a  | &   a  | &   a   |  
#          |  2   1 |  2   2 |  2   3  |
#          |        |        |         |
#          |  2   1 |  2   1 |  2   1  |
#          | &   a  | &   a  | &   a   |
#          |  3   1 |  3   2 |  3   3  |
#          |________|________|_________|


# Once we are done we will have a matrix of partial derivatives, and this tells us how each of these weights affect
#the total cost, if we are given a vector of all the errors in our neural network, and if we are given all the
#activations in our neural network.

# Backpropagation Equations

#     L    ___          1  L
# 1. &  =  \ / a C (.) r (Z )

#     L       L + 1  T  L + 1       1   1
# 2. &  =  ((W     )   &     ) (.) r  (Z )

#   2C        L
# ------  =  &
#   2b(L)

#   2C        L   L - 1 T
# ------  =  &  (a     )
#   2W(L)

# These are the steps in which we will use these equations while performing backpropagatioon.

# 1. Forward propagation we will compute the a(L)'s and the Z(1)'s.

# 2. Compute Cost

# 3. Use equation 1 to calculate our first layer of errors.

# 4. Using equations 1 and 2 we we can backpropagate and calculate every error in our neural network.

# 5. Use equation 3 to find the cost to the bias derivatives.

# 6. Use equation 4 to calculate weight derivatives.


# Using these 6 rules of backpropagation we can calculate the gradient of the cost with respect to all weights
#and biases.

# 