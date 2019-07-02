#%% Initialization
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import tensorflow as tf
import timeit
from numpy import linalg

#%% Parameters
a,b,g,s = 1.0,0.1,1.5,0.75
ts = 2000
lags = 8
neurons = 20
batch_size = 128
iterations = 20000

#%% Lotka-Volterra Equations
def dzdt(z,t):
    m,n = z
    dxdt = (a*m)- (b*m*n)
    dydt = -(g*n) + (s*m*n)
    return [dxdt, dydt]

# lags
def lags_data(dat, lags):
    n_dat = len(dat)-lags
    d_dat = dat.shape[1]
    y = np.zeros((lags, n_dat, d_dat))
    yt = np.zeros((n_dat, d_dat))
    for i in range(0,n_dat):
        y[:,i,:] = dat[i:(i+lags), :]
        yt[i,:] = dat[i + lags, :]
    return y, yt



#%% LSTM class
    
class LSTM:
    # Initialize the class
    def __init__(self, X, Y, hidden_dim):
        
        # X has the form lags x data x dim
        # Y has the form data x dim
     
        self.X = X
        self.Y = Y
        
        self.X_dim = X.shape[-1]
        self.Y_dim = Y.shape[-1]
        self.hidden_dim = hidden_dim
        self.lags = X.shape[0]

        # Initialize network weights and biases        
        self.Wf, self.bf, self.Wi, self.bi, self.Ws, self.bs, self.Wo, self.bo, self.V, self.c = self.initialize_LSTM()
                
        # Store loss values
        self.training_loss = [] 
        
        # Define Tensorflow session
        self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        
        # Define placeholders and computational graph
        self.X_tf = tf.placeholder(tf.float32, shape=(self.X.shape[0], None, self.X.shape[2]))
        self.Y_tf = tf.placeholder(tf.float32, shape=(None, self.Y.shape[1]))
        
        # Evaluate prediction
        self.Y_pred = self.forward_pass(self.X_tf)
        
        # Evaluate loss
        self.loss = tf.losses.mean_squared_error(self.Y_tf, self.Y_pred)
        
        # Define optimizer        
        self.optimizer = tf.train.AdamOptimizer(1e-3)
        self.train_op = self.optimizer.minimize(self.loss)
        
        # Initialize Tensorflow variables
        init = tf.global_variables_initializer()
        self.sess.run(init)

    
    # Initialize network weights and biases using Xavier initialization
    def initialize_LSTM(self):      
        # Xavier initialization
        def xavier_init(size):
            in_dim = size[0]
            out_dim = size[1]
            xavier_stddev = 1. / np.sqrt((in_dim + out_dim) / 2.)
            return tf.Variable(tf.random_normal([in_dim, out_dim], dtype=tf.float32) * xavier_stddev, dtype=tf.float32)    
        
        Wf = xavier_init(size=[self.X_dim, self.hidden_dim])
        bf = tf.Variable(tf.zeros([1,self.hidden_dim], dtype=tf.float32), dtype=tf.float32)
        
        Wi = xavier_init(size=[self.X_dim, self.hidden_dim])
        bi = tf.Variable(tf.zeros([1,self.hidden_dim], dtype=tf.float32), dtype=tf.float32)
        
        Ws = xavier_init(size=[self.X_dim, self.hidden_dim])
        bs = tf.Variable(tf.zeros([1,self.hidden_dim], dtype=tf.float32), dtype=tf.float32)

        Wo = xavier_init(size=[self.X_dim, self.hidden_dim])
        bo = tf.Variable(tf.zeros([1,self.hidden_dim], dtype=tf.float32), dtype=tf.float32)    
            
        V = xavier_init(size=[self.hidden_dim, self.Y_dim])
        c = tf.Variable(tf.zeros([1,self.Y_dim], dtype=tf.float32), dtype=tf.float32)
            
        return Wf, bf, Wi, bi, Ws, bs, Wo, bo, V, c
    
    def forward_pass(self, X):
        H = tf.zeros([tf.shape(X)[1], self.hidden_dim], dtype=tf.float32)
        st_l = 0
        for i in range(0, lags):
            ft = tf.nn.sigmoid(tf.add(tf.matmul(X[i,:,:],self.Wf), self.bf))
            it = tf.nn.sigmoid(tf.add(tf.matmul(X[i,:,:],self.Wi), self.bi))
            scap = tf.nn.tanh(tf.add(tf.matmul(X[i,:,:],self.Ws), self.bs))
            ot = tf.nn.sigmoid(tf.add(tf.matmul(X[i,:,:],self.Wo), self.bo))
            st = tf.math.multiply(ft,st_l) + tf.math.multiply(it,scap)
            st_l = st
            H = tf.math.multiply(ot,tf.nn.tanh(st))   
        H = tf.add(tf.matmul(H,self.V),self.c)
        return H
    
    
    # Fetches a mini-batch of data
    def fetch_minibatch(self,X, Y, N_batch):
        N = X.shape[1]
        idx = np.random.choice(N, N_batch, replace=False)
        X_batch = X[:,idx,:]
        Y_batch = Y[idx,:]        
        return X_batch, Y_batch
    
    
    # Trains the model by minimizing the MSE loss
    def train(self, nIter , batch_size ): 

        start_time = timeit.default_timer()
        for it in range(nIter):     
            # Fetch a mini-batch of data
            X_batch, Y_batch = self.fetch_minibatch(self.X, self.Y, batch_size)
            
            # Define a dictionary for associating placeholders with data
            tf_dict = {self.X_tf: X_batch, self.Y_tf: Y_batch}  
            
            # Run the Tensorflow session to minimize the loss
            self.sess.run(self.train_op, tf_dict)

            
            # Print
            if it % 1000 == 0:
                elapsed = timeit.default_timer() - start_time
                loss_value = self.sess.run(self.loss, tf_dict)
                print('It: %d, Loss: %.3e, Time: %.2f' % 
                      (it, loss_value, elapsed))
                start_time = timeit.default_timer()
                
                
    # Evaluates predictions at test points           
    def predict(self, X_star):      
        tf_dict = {self.X_tf: X_star}       
        Y_star = self.sess.run(self.Y_pred, tf_dict) 
        return Y_star
    
#%% Integration

zinit = np.array([10,5])
t = np.linspace(0,60,ts)
z = integrate.odeint(dzdt,zinit,t)
t = np.reshape(t,[np.size(t,0),1])

#%% Preparing data and getting lags
# normalizing data   
zf = (z - np.mean(z, axis=0, keepdims=True))/ np.std(z, axis=0, keepdims=True)

# Select first 2/3 of data for training
zt = zf[0:int(len(zf) * (2.0/3.0)),:]

X, Y = lags_data(zt, lags)
     
#%% Training
model = LSTM(X, Y, neurons)
model.train(iterations, batch_size)

#%% Prediction
pred = np.zeros((len(zf)-lags, Y.shape[-1]))
X_tmp =  np.copy(X[:,0:1,:])
for i in range(0, len(zf)-lags):
    pred[i] = model.predict(X_tmp)
    X_tmp[:-1,:,:] = np.copy(X_tmp[1:,:,:]) 
    X_tmp[-1,:,:] = np.copy(pred[i])
    
#%% Calculating L2 Norm
xL2 = linalg.norm(zf[lags:,0:1]-pred[0:1], 2)/linalg.norm(zf[lags:,0:1], 2)
yL2 = linalg.norm(zf[lags:,1:2]-pred[1:2], 2)/linalg.norm(zf[lags:,1:2], 2)
print('')
print('Relative ||L||_{2} norm (Rabbits): ', xL2)
print('Relative ||L||_{2} norm (Foxes): ', yL2)

#%% Plotting

plt.figure(1)
plt.plot(t[lags:,0],zf[lags:,0], 'b-', linewidth = 2, label = "Exact")
plt.plot(t[lags:,0],pred[:,0], 'r--', linewidth = 3, label = "Prediction")
plt.plot(t[X.shape[1]]*np.ones((2,1)), np.linspace(-1.75,1.75,2), 'k--', linewidth=2)
plt.axis('tight')
plt.xlabel('$t$')
plt.ylabel('$x_t$')
plt.legend(loc='lower left')

plt.figure(2)
plt.plot(t[lags:,0],zf[lags:,1], 'b-', linewidth = 2, label = "Exact")
plt.plot(t[lags:,0],pred[:,1], 'r--', linewidth = 3, label = "Prediction")
plt.plot(t[X.shape[1]]*np.ones((2,1)), np.linspace(-1.75,1.75,2), 'k--', linewidth=2)
plt.axis('tight')
plt.xlabel('$t$')
plt.ylabel('$y_t$')
plt.legend(loc='lower left')

