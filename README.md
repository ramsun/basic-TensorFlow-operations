```
import tensorflow as tf
import keras
import numpy as np
```


```
tf_version = tf.__version__
keras_version = keras.__version__
print("current tensorflow version: ",tf_version)
print("current keras version: ",keras_version)
```

    current tensorflow version:  2.3.0
    current keras version:  2.4.3


# Basic TensorFlow Operations
1. Declaration
2. Stacking
3. Reshaping
4. Scalar operations
5. Associative and Distributive properties



```
# Basic constant tensor declaration and addition

# Make sure to define type as int64 to perform math calculation. Default = int32
x = tf.constant([100,101,102,103,104,105,106,107,108,109], dtype = 'int64')
# tensor can also accept numpy arrays
numpy_array = np.array([34,28,45,67,89,93,24,49,11,7], dtype = 'int64')
y = tf.constant(numpy_array)

# Perform sum and output to notebook window
tensor_sum = tf.add(x, y)

print("Sum of tensors is: ", tensor_sum)
```

    Sum of tensors is:  tf.Tensor([134 129 147 170 193 198 130 156 119 116], shape=(10,), dtype=int64)


Shape has a weird notation in tensorflow.  Here are examples of what the shapes mean in 1D, 2D, and 3D:
1. shape=(6,) in TF imlpies row vector of size 1 by 6 (1 row and 6 columns).
2. shape=(2,6) in TF implies 2D tensor with 2 rows and 6 columns.
3. shape=(3,2,6) implies a 3D matrix with 2 rows, 6 columns, and depth (z axis) of 3 (standard notation would be 2x6x3).
 


```
# Reshape a tensor with the stack command

# Stack differs from concatonation since it creates a new axis for each stack 
# (stacks in the z direction for a 2D tensor, thus creating a new dimension)
# If our tensor were 1D, stack would create a 2D tensor
x1 = tf.constant([[1,2,3,4],[5,6,7,8]])
x1_stack = tf.stack([x1,x1,x1,x1], axis=0)

print(x1_stack)
```

    tf.Tensor(
    [[[1 2 3 4]
      [5 6 7 8]]
    
     [[1 2 3 4]
      [5 6 7 8]]
    
     [[1 2 3 4]
      [5 6 7 8]]
    
     [[1 2 3 4]
      [5 6 7 8]]], shape=(4, 2, 4), dtype=int32)



```
# stacking a 2D tensor by itself makes TF think of it as a 3 dimensional tensor
# It adds a new axis
x1 = tf.constant([[1,2,3,4],[5,6,7,8]])
x1_stack = tf.stack([x1], axis=0)

print(x1_stack)
```

    tf.Tensor(
    [[[1 2 3 4]
      [5 6 7 8]]], shape=(1, 2, 4), dtype=int32)



```
# Reshape a tensor

# Suppose ‘x1’ a tensor of shape (3,4).
# reshape it into shape (6,2)
x1 = tf.constant([[1,2,3,4],[5,6,7,8],[9,10,11,12]])

# Reshape method parameters:
# input the tensor and then new shape you want to resize to
# new shape needs to hold the same number of elements as the original shape
t_reshaped = tf.reshape(x1, [6,2])

print(t_reshaped)
```

    tf.Tensor(
    [[ 1  2]
     [ 3  4]
     [ 5  6]
     [ 7  8]
     [ 9 10]
     [11 12]], shape=(6, 2), dtype=int32)



```
# Operations with scalar tensors

# Define rank 0 tensors
a = tf.constant(1.12)
b = tf.constant(2.34)
c = tf.constant(0.72)
d = tf.constant(0.81)
e = tf.constant(19.83)

# We can perform calculation by simply using *,/,+,- operators
# These operators simply call tf.add() and the like, so they are identical
# Each of these operators are treated as element-wise operations on scalars,
# which are simply rank 0 tensors in this case
x = 1 + a/b + c/(e*e)
s = (b-a) / (d-c)
r = 1/((1/a)+(1/b)+(1/c)+(1/d))
y = a*b * (1/c) * (e*e/2)

print("x: " , x, "s: ", s, "r: ", r, "y: ", y)
```

    x:  tf.Tensor(1.4804634, shape=(), dtype=float32) s:  tf.Tensor(13.555558, shape=(), dtype=float32) r:  tf.Tensor(0.25357127, shape=(), dtype=float32) y:  tf.Tensor(715.6765, shape=(), dtype=float32)



```
#Associate and Distributive properties (Avoid Hadamard product)

# Define three tensors
A = tf.constant([[4,-2,1],[6,8,-5],[7,9,10]])
B = tf.constant([[6,9,-4],[7,5,3],[-8,2,1]])
C = ([-4,-5,2],[10,6,1],[3,-9,8])

# @ symbol simply calls matmul as of python >= 3.5
associative_property_LHS = A @ (B + C) 
associative_property_RHS = A @ B + A @ C
distributive_property_LHS = (A @ B) @ C
distributive_property_RHS = A @ (B @ C)

associative_property_boolean = associative_property_LHS == associative_property_RHS
distributive_property_boolean = distributive_property_LHS == distributive_property_RHS

print("Associative Property:", associative_property_boolean)
print("Distributive Property:", distributive_property_boolean)
```

    Associative Property: tf.Tensor(
    [[ True  True  True]
     [ True  True  True]
     [ True  True  True]], shape=(3, 3), dtype=bool)
    Distributive Property: tf.Tensor(
    [[ True  True  True]
     [ True  True  True]
     [ True  True  True]], shape=(3, 3), dtype=bool)

