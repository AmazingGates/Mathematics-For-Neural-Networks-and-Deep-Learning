# In this section we will be going over all of the aspects of Forward Propagagtion.


# The first thing we will discuss is the neuron function

#   Neuron 

# Inputs are dot products
# Example

# First, we're going to describe how we can mathematically explore the concept of a single neuron in the 
#neural network.

# So let's dive in with an example. 

# X1\W1
# X2 \W2
# X3 W3-------> N
# X4 /W4
# X5/W5

# These will be the inputs to a single neural, (N).

# All of these inputs are connected to this neuron by some weight.

# What our neuron will do is, it will take the weighted sum of all the inputs.

# So it's going to multiply an input by its corresponding weight. 

# For example, we'd have X1 times W1 plus  X2 times W2 plus X3 times W3 etc etc.

# To describe this we can say, the summation of the i = 1, up until n (n equals the amount of input variables we have),
#and then we'll do Xi times Wi. Once all of the inputs and their corresponding weights are multiplied and summed up,
#then we will add our bias.

# This is how we describe what we are doing mathematically.

# We sum over all of our inputs, we go from 1 to n, and then we multiply the first input by the first weight, and 
#then we add that to the second input times the second weight, etc etc.

# This will leave us with a single scalar.

# Then we have our bias. And the bias is a term that is attach to the node / neuron. It's a scalar that we add to 
#the node / neuron.

# We can explain all of this in a simplified manner using dot products.


#   Inputs are dot products -
# -  We will start talking about the dot product of two vectors. See Example below

# Let's say we have two vectors __ __  __ __
#                              |  3  ||  4  |
#                              |  2  ||  5  |
#                              |__1__||__6__|

# The dot product specifies that we will transpose one of the vectors, and multiply it by the other.

# So we end up with something like this.

#                        __ __
#                       |  4  |
#                       |  5  |
#               [3 2 1] |__6__|

# To operate this equation, we multiply 3 by 4 plus 2 time 5 plus 1 times 6.

# This will give us a scalar output 28 from this dot product.

# The important part of dot product is that we multiply to vectors and end up with a scalar, sometimes called the
#inner product.

# That's a simple review on what a dot product is.

# We can correlate this dot product to our inputs and weights.

# We can separate the inputs and weights into two vectors and use the dot product on them.

# That will end up looking like this, see example below.

# X1W1 + X2W2 + X3W3 + X4W4 + X5W5

# We will think about it, that's exactly what the summation is doing, so we replace it with some simple notation 
#for dot product, which will give us something like this.

# ( Xt W + b )

# Remember that when using the dot product one of the vectors has to be transposed, and here we specified that by 
#using the letter t next to the X to indicate that we want the input vector to be transposed.

# This is a simplified notation for a single neuron using dot product.

# Another key idea to essentially know is that this ( Xt W + b ), can be known as the weighted sum, and is equal to z.

# We can actually represnt this neuron in two steps. 

# z = ( Xt W + b ) is the first part.

# Our second part can be putting it into a sigmoid or whatever activation function we are using. So the activation of
#this neuron is going to a. 

# a = activation function (z). Note, we only used the phrase activation function as a place holder. When writing
#live code we'd use the actual activation we're going to be using.

# The key idea we want to get across is that our single neuron takes in a vector and outputs a scalar. 

# So it's a vector to scalar function.

# It basically inputs a vector, and then we get a scalar.



#   Weight and Bias Indexing

# Next we will look at the layers of neurons and how we can represent them with entire matrices.

# The first thing we will do is talk about the notation of the weights in our neural network.

# So let's say that we have an input of X1 and X2, then two layers of neurons, and finally our output.

# X1  O  O
#   \/ \/ \ O
#   /\ /\ / 
# X2  O  O

# Before, we would just use W1, W2, W3 etc etc to represent the weights that connect our inputs to the neurons, but
#now that we have multiple nodes that wouldn't be the proper method.

# So basically what we would do is split this up into another notation, W(l, j, k)

# So what does this mean?

# This is going to be the standard notation for any single weight in our neural network.

# L equals the layer the weight is going into.

# J equals/specifes which node the weight is going into in layer L.

# K wants us to specify the node in layer L minus 1. Basically K wants to know from which node in the originating layer
#did the weight come from.

# Now let's look at an example that will make things a little more understanable.

# Let's say we want to establish what the weight marked example 1 is

# X1__O__O
#   \/ \/ \ O
#   /\ /\ / 
# X2__O__O
#       |
#       Example 1

# So we can first see that the weight is coming from layer 1, and it is going into layer 2.

# Also we can note that the weight is going into the second node of layer 2.

# Now we will specify that the node the weight came from in layer 1 is the second node.

# So we can use this information to substitute our letters with their corresponding values.

# W(L=2, J=2, K=2)

# Let's look at one more example to further understand.

# This time we will establish what the weight marked example is.

# X1__O__O
#   \/ \/ \--Example 2
#   /\ /\ / O (Note that this output is between the weights, not alongside them.)
# X2__O__O

# First we can see that the weight is going into layer 3(output layer).

# Second we can see that the weight is going into the output node.

# Finally, we can see that the node came from the first node in the originating layer.

# W(3, 1, 1)

# Now we want to cive into the biases of this example.

# X1__O__O
#   \/ \/ \ O
#   /\ /\ / 
# X2__O__O

# So generally biases are the same for each node in a layer.

# So for example, layer one may have a bias of 5, layer two may have a bias of 8, and the third layer/output layer
#may have a bias of 3.

# Generally, the bias for 1 layer is signified by b(L)

# So for example b(1) would equal 5, since we specified that the bias for layer one was 5.

# And that goes for the rest of the layers.

# That is generally speaking.

# There are instances where we might see something like b(L, J). This indicates that each node in the layer has 
#its own bias.

# Attaching biases to layers and not individual nodes is a lot more common.

# But for example, if we did see a layer with nodes that had their own biases, something like node 1 in layer one is 3,
#and node 2 in layer one is 5, it would be represented like this, b(1, 1) equals 3, and b(1, 2) equals 5.

# This can be read as the bias for the first layers first node equals 3, and the bias for the first layers second node
#equals 5, for example.

# Moving forward, we will be dealing with situations where the bias is for the entire layer, node individual nodes.

# Now we will dive into matrices and how they actually operate.



#   A Layer of Neurons

# Now that we have the notation down we can jump into how we can represent the computations for an entire layer
#of the neural network.

# When we're thinking of large networks of neurons, let's say we have three neurons for example, and we have five
#inputs.

# Each of the five inputs would have a connection with each of the neurons. See example below

# X1

# X2        O

# X3        O

# X4        O

# X5

# Image X1 having three connections. One to every neuron. 

# The same would be the same for all of the other inputs.

# Those connections that go from the inputs into the neurons is called a weight.

# So each neuron is recieving a weighted sum from the input and its weight.

# Basically, each node is recieving five weighted inputs that are its own.

# So since we saw that a single node takes in a vector of inputs, and then multiplies it by a vector of weights,
#and then puts out a single scalar, we can represent this something like this.

# X1

# X2        O --> S1

# X3        O --> S2

# X4        O --> S3

# X5

# Those scalars would form a new vector of inputs, just like the X inputs, that go into our next layer of neural 
#network.

# So we send each of these, S1, S2, S3, into some neuron, and then we're going to have the same thing.

# So the scalars that are outputed from the neurons, are actually going to become a vector, and this vector is 
#what we like to call A(L), which is the activations of layer L.

# Since it's coming from layer one, we can call it A(1), to represent the activations of layer 1.

# Now, how can we represent this?

# Remember that we represent a single node like this O(X t W + b), which is X transpose W plus b.

# This is how we represent our single node, where X equals a vector, W equals a Vector, which is the scalar, and b
#is the bias. Then we put everything through an activation.

# Well now, we can represent the weights of an entire layer with a weights matrix.

# So that can represented with a capital W, we usually use capital letters to specify matrices, W(L).

# So this takes care of all the weights in an entire layer, for example, that go from each input to each node.

# So how do we do that?

# One thing to remember about the weights matrix, is to remember that to locate a single weight in an entire network, 
#we can use this formula, W(L,J,K), where J equals the number of the neuron in the layer the weight is going to,
#and K equals the number of the neuron in the layer where the weight came from. 

# So when we're looking at the weights matrix, each entry in the weights matrix gets entered like this.

# The Rows are the (J), and the columns are the (k), and the entire matrix is the layer L, the layer in which the 
#weight is going.

# For example let's say that we have this neuron in our matrix. With a label of 1 for K, and a label of 1 for J.

# This would indicate that the weight came from the first neuron in the previous layer, and went into the first 
#neuron in the new layer. 

#        _____ K _____
#       |      1      |
#       |   o         |
# W =  J|1            |
#       |             |
#       |             |
#       |_____   _____|

# Note: In general, our weights matrices are (n,m), where n - nodes in layer l, and m = nodes in layer l minus one.

# Let's look at another example below, using this simple neural network. 

# X1 - O\
#    /\  O
# X2 - O/

# We will give all of weights values.

# Input 1 to neuron 1 will be 5.
# Inout 1 to neuron 2 will be 3
# Input 2 to neuron 1 will be 2
# Input 2 to neuron 2 will be 6

# The K is the node the weight is coming from in the previous layer.

# So if it was coming from the first input we would indicate that in the first column.
# If it was coming from the second input we would indicate that in the second column.

# And if it was going into the first node of the new layer, we would indicate that in the first row.
# If it was going into the second node of the new layer, we would indicate that in the second row.

# That is Basically how we locate something in the weighted matrices.

# Note: We use the method of (r,c) to indicate which weight we are locating the value for.
# Also note that inputs represent (K), because this is the where the weight is coming from, and (K) represents the
#column in our matrix

# So let's look at the first example of (1,1) again.

# Notice that we have a value of 5 in our matrix.

# This is because if we remember, input 1 represents where the weight came from inside of the column, and neuron 1 
#represents what neuron in the new layer the weight goes into, which will have a weighted value of 5.

# Next we will look at the location (1, 2)

# Notice that we have a value of 2 in our matrix.

# This is because if we remember, input 2 represents where the weight came from inside of the column, and neuron 1
#represents what neuron in the new layer the weight goes into, which will have a weighted value of 2.

# Next we will look at the location (2, 1)

# Notice that we have a value of 3 in our matrix.

# This is because if we remember, input 1 represents where the weight came from inside of the column, and neuron 2
#represents what neuron in the new layer the weight goes into, which will have a weighted value of 3.

# Next we will look at the location (2, 2)

# Notice that we have a value of 6 in our matrix.

# This is because if we remember, input 2 represents where the weight came from inside of the column, and neuron 2
#represents what neuron in the new layer the weight goes into, which will have a weighted value of 6.


#        _____ K _____
#       |             |
#       |  5       2  |
# W =  J|             |
#       |             |
#       |  3       6  |
#       |_____   _____|

# So this is the weights matrix.

# Now we're going to draw out the notation for the entire layer, just like we did for the single neuron.

# r( W(L) times a(L-1) + b(L) )

# Now let's actually work the equation to see why it works.

# First, the a(L-1) is the activation of the previous layer.

# Recall that when we have a whole layer of neurons, the scalar outputs become the vector, that vector would be
#an a vector.

# So for the first layer, our A, would be the X inputs, but for the second layer the A would be the output of the first 
#layer, and so and so forth.

# A lot of the time we represent the X input as either a zero or something like that, because it's the first
#activation.

# So back to our calculation and why it works.

# Let's say our inputs are 10 and 20, for example purposes.

# So those are our inputs.

# So our formula tells us to multiply the weights matrix by the activations of the previous layer, W(L) times a(L-1).

# We Can visualize this process like this.


#        _____ K _____    ____ ____
#       |             |  |         |
#       |  5       2  |  | 10 = X1 |
# W =  J|             |  |         | = Activations of Previous Layer
#       |             |  |         |
#       |  3       6  |  | 20 = X2 |
#       |_____   _____|  |____ ____|
#            (2, 2)        (2, 1)


# To do matrix multiplication, first we understand that the weights matrix is a 2 by 2, and the Activations of 
#previous layer is a 2 by 1.

# This will give us an output vector of 2 by 1 once we are done calculating.

# The first calculations we will do are the 5 and 2 weight values.

# Since 5 is the weight value connected to input 1, it gets multiplied by 10.

# Since 2 is the weight value connected to input 2, it gets multiplied by 20.

# Calculating these values will give us the weighted sum going into that neuron.

# Now let's calulate.

# 5 times 10 = 50

# 2 times 20 = 40

# 50 + 40 = 90

# This means that 90 is the weighted sum of that neuron.

# Next we will do the second row of values, which are 3 and 6.

# 3 times 10 = 30

# 6 times 20 = 120

# 30 + 120 = 150

# So the weighted sum of that neuron is 150

# NOTE: These are the values before we add the biases.

#   __ __
#  |     |
#  |  90 |
#  |     |
#  | 150 |   
#  |__ __|

# Now we will add our biases.

# Usually the bias is the same for the entire layer, unless specified as otherwise.

# Let's say our bias here is 6, for example.

# That means we will add 6 to each of  our results and get our final solution.

#   __ __
#  |     |
#  |  96 |
#  |     |
#  | 156 |   
#  |__ __|

# So now we've calculated the entire inside, W(L) times a(L-1) + b(L), which we usually call Z.

# Now we can say 

#         __ __
#        |     |
#        |  96 |
# Z(1) = |     |
#        | 156 |   
#        |__ __|

# We still have one more step before we are done. We have the activation function (r).

# We have to put the Z(1) into an activation function.

#         (__ __)
#        (|     |)
#        (|  96 |)
#     R  (|     |)
#        (| 156 |)   
#        (|__ __|)

# After going through the activation process, we can say that this is the scalar output for that layer.

# Now we can say that

#         __ __
#        |     |
#        |  96 |
# A(1) = |     |
#        | 156 |   
#        |__ __|

# This indicates that this is the complete and final Activation for the Previous Layer when we move on to the next layer.

# Another way we can represent multi layers in a neural network in a much simplier fashion is to use a chain of
#multiplications (see example below).

# First we have our inputs, let's call them A0. 

# Then we put them into our (z)

# This calculation calculates the entire first layer, the output for the entire first layer of our network.

# A0 => r(W1 times A0 + b1)

# Remeber that we are still dealing with single training examples, so our input A0 is a vector.

# So that vector gets fed into our Z, and the output is another vetcor.

# This is because if we have multiple neurons, each of those is going to set out a scalar and become a vector.

# So this, A0 => r(W1 times A0 + b1), takes in a vector and outputs a vector.

# So the output of this, r(W1 times A0 + b1), is A1.

# So no we can put this, r(W1 times A0 + b1) =>, into another layer of the network and get this new equation

#A0 => r(W1 times A0 + b1) => r(W2 times A1 + b2)

# This, r(W2 times A1 + b2), is the second layer of our neural network, and we could just keep goiing for however
#many layers we have in our neural network.

# This, A0 => r(W1 times A0 + b1) => r(W2 times A1 + b2), is a much simplier way of computing neural networks.

# You can think of this as the feed forward method.