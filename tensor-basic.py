
import tensorflow as tf
import os
if __name__ == "__main__":
    physical_devices=tf.config.list_physical_devices("GPU")
    tf.config.experimental.set_memory_growth(physical_devices[0],True)
    #initialize the matrix of 4
    x = tf.constant(4,shape=(10,10),dtype=tf.float32)
    x=tf.constant([[1,2,3],[4,5,6]])
    #init the matrix of ones and zeros
    ones=tf.ones(shape=(4,4))
    zeros=tf.zeros(shape=(4,4))

    rand_dist=tf.random.normal((3,3),mean=0,stddev=1)

    unifor_rand=tf.random.uniform((1,10),minval=0,maxval=1)
    #delta is the stepsize here
    __range=tf.range(start=1,limit=10,delta=2)
    #type conversition  tf.float(16,32,64) tf.int(8,16,32,64,) tf.bool

    __range=tf.cast(__range,dtype=tf.float64)
        
    #operation 
    x=tf.constant([[1,2,3]])
    y=tf.constant([[9,8,7]])
    x1=tf.random.normal((3,3),mean=0,stddev=2)
    x2=tf.random.normal((3,3),mean=0,stddev=1)
   # print(f"Element wise Sumation{tf.add(x,y)}")
    dot_prod=tf.tensordot(x1,x2,axes=1)
    #print(f"Elementwise dot product{dot_prod}")
    matrix_mul=tf.matmul(x1,x2)
    #print(f"matrix multiplication {matrix_mul}")

    #indexing :: for jump and - for reversing the  vector or matrix

    x=tf.constant([1,2,3,4,5,6])
    #print(f"reverse the elements{x[::-1]}")
    indices=tf.constant([0,3])
    x_inde=tf.gather(x,indices)
    print(f"indexing the vector{x_inde}")
    x=tf.constant([[1,2,3,4],[5,6,4,8],[7,8,9,10]])
    print(f"all the eements{x[0,:]}")




