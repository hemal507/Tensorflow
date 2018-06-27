import tensorflow as tf

# Creating variable for parameter slope (W) with initial value as 0.4
W = tf.Variable([.4], tf.float32)
 
#Creating variable for parameter bias (b) with initial value as -0.4
b = tf.Variable([-0.4], tf.float32)
 
# Creating placeholders for providing input or independent variable, denoted by x
x = tf.placeholder(tf.float32)
 
# Equation of Linear Regression
linear_model = W * x + b

# Initializing all the variables
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

# part 2 
# Running regression model to calculate the output w.r.t. to provided x values
# print(sess.run(linear_model, {x: [1, 2, 3, 4]})) 

y = tf.placeholder(tf.float32)
error = linear_model - y
squared_errors = tf.square(error)
loss = tf.reduce_sum(squared_errors)
# print(sess.run(loss, {x:[1,2,3,4], y:[2, 4, 6, 8]}))


# part 3
# Creating an instance of gradient descent optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
 
train = optimizer.minimize(loss)
 
for i in range(1000):
     sess.run(train, {x:[1, 2, 3, 4], y:[2, 4, 6, 8]})
print(sess.run([W, b]))
