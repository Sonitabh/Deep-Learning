#%% Initialization
from pyDOE import lhs
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


#%% Parameters

eta = 0.001
Nmin = 100
Nmax = 5000
step = 100
iteration = 10000

print('#######START#######')
print('')    
print('Parameters :')
print('Learning rate = ',eta)
print('Number of training points range from',Nmin,'to',Nmax, 'at steps of', step)
print('Number of iterations for each training step is',iteration)
print('')



#%% Looping through for different number of training points
Nval = []
L2val = []

for j in range(Nmin,Nmax+1,step):
    Nval.append(j)
    print('Number of Training points :',j)
#%% Generate data
    x = 4*lhs(1,j) + 50
    y = 4*lhs(1,j) + 50
    f = np.cos(np.pi*x)*np.sin(np.pi*y)
    
    A = np.hstack((x,y))                # creating a single input matrix for both variables
    
    scaler = StandardScaler()           # Batch normalization
    A = scaler.fit_transform(A)
    
    # Train-Test split
    A, A_star, f, f_star = train_test_split(A, f, test_size=0.2, random_state = 42)
    
    # Defining layer dimentions
    lay =[2,50,50,1]
    
    
    #%% Creating NN Computational graph
    
    # Define NN
    def NN(A,lay):
        
        # Normal Xavier paremeter initialization
        var1 = tf.divide(2,(lay[0]+lay[1]))
        w1 = tf.Variable(tf.random_normal([lay[0],lay[1]],mean=0.0,stddev = tf.sqrt(var1),dtype=tf.float32,seed=1))
        b1 = tf.Variable(tf.zeros([lay[1]]))
        
        var2 = tf.divide(2,(lay[1]+lay[2]))    
        w2 = tf.Variable(tf.random_normal([lay[1],lay[2]],mean=0.0,stddev = tf.sqrt(var2),dtype=tf.float32,seed=1))
        b2 = tf.Variable(tf.zeros([lay[2]]))
        
        var3 = tf.divide(2,(lay[2]+lay[3]))    
        w3 = tf.Variable(tf.random_normal([lay[2],lay[3]],mean=0.0,stddev = tf.sqrt(var3),dtype=tf.float32,seed=1))
        b3 = tf.Variable(tf.zeros([lay[3]]))
        
        # Making the forward pass
        L1 = tf.add(tf.matmul(A,w1), b1)
        H1 = tf.nn.tanh(L1)
    
        L2 = tf.add(tf.matmul(H1,w2), b2)
        H2 = tf.nn.tanh(L2)
    
        L3 = tf.add(tf.matmul(H2,w3), b3)    
        
        return L3
    
    # Placeholders
    A_p = tf.placeholder(tf.float32)
    f_p = tf.placeholder(tf.float32)
    
    # Getting the output
    Out = NN(A_p,lay)
    
    
    # loss
    mse = tf.reduce_mean(tf.square(Out-f_p))
    
    # L2
    L2_loss =  tf.divide(tf.sqrt(tf.matmul(tf.transpose(f_p-Out),(f_p-Out))),tf.sqrt(tf.matmul(tf.transpose(f_p),f_p)))
    
    
    # Choose optimizer
    opti = tf.train.AdamOptimizer(eta).minimize(mse)
    
    # Storing training error
    train_err = []
    
    #%% Session
    
    init = tf.global_variables_initializer()
    save1 = tf.train.Saver()
    S = tf.Session()
    S.run(init)
    
    for k in range(iteration):
        S.run([opti],feed_dict = {A_p: A, f_p:f})
        
        train_err.append(S.run(mse, feed_dict={A_p:A,f_p:f}))
        '''   
        if k%(500) == 0:
            print('Iteration:',k)
            print('Error :',train_err[k])
            print('')
        '''   
    # Predictions
    pred = S.run(Out, feed_dict={A_p:A_star})
    L2_loss = S.run(L2_loss, feed_dict={A_p:A_star,f_p:f_star})
    print('L2 Approximation Error :',L2_loss[0][0])
    print('')
    L2val.append(L2_loss)
    
    tf.Session.close

L2val = np.asarray(L2val)
L2val = np.reshape(L2val,(L2val.shape[0],1))
#%% Plots

# 45 degree plot
plt.figure(1)
plt.plot(f_star,pred,'ro',markersize=2,label = 'Predicted data points')
plt.plot(np.linspace(np.min(f_star), np.max(f_star)),np.linspace(np.min(f_star), np.max(f_star)),'b-',label = '45 degree line')
plt.title('45 degree plot')
plt.ylabel('Predicted Data')
plt.xlabel('Original Data')
plt.legend()

# MSE Train error
plt.figure(2)
plt.plot(train_err, 'r',linewidth=3.0, label = 'Training Error')
plt.title('Training Error (MSE)')
plt.xlabel('$Iterations$')
plt.ylabel('$MSE$ $Loss$')
plt.show()

# L2 vs Number of points
plt.figure(3)
plt.plot(Nval, L2val)
plt.title('||L||$_{2}$ Approximation Error')
plt.xlabel('Iterations')
plt.ylabel('||L||$_{2}$')

