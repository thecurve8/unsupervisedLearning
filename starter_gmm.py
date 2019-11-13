import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import helper as hlp
import starter_kmeans as km
import pickle 
# Loading data
#data = np.load('data100D.npy')
data = np.load('data2D.npy')
[num_pts, dim] = np.shape(data)

# Distance function for GMM
def distanceFunc(X, MU):
    # Inputs
    # X: is an NxD matrix (N observations and D dimensions)
    # MU: is an KxD matrix (K means and D dimensions)
    # Outputs
    # pair_dist: is the pairwise distance matrix (NxK)
    expand_x = np.expand_dims(X, 1)
    expand_mu = np.expand_dims(MU, 0)
    pair_dist = np.sum(np.square(np.subtract(expand_x, expand_mu)), 2)
    return pair_dist

def log_GaussPDF(X, mu, sigma):
    # Inputs
    # X: N X D
    # mu: K X D
    # sigma: K X 1

    # Outputs:
    # log Gaussian PDF N X K
    # [...log(N(x_n, mu_k, sigma_k))...]
    pi = 3.14159
    expand_x = tf.expand_dims(X, 0)
    expand_mu = tf.expand_dims(mu, 1)
    pair_dist = tf.reduce_sum(np.square(tf.subtract(expand_x, expand_mu)), 2)
    exponent = tf.divide(pair_dist, -2*sigma)
    coef = 1/(tf.pow(2*pi*sigma, dim/2))
    return tf.transpose(tf.log(coef)+exponent)


def log_posterior(log_PDF, log_pi):
    # Input
    # log_PDF: log Gaussian PDF N X K
    # log_pi: K X 1

    # Outputs
    # log_post: N X K
    log_density = log_PDF + tf.transpose(log_pi)
    log_sums = hlp.reduce_logsumexp(log_density)
    log_sums = tf.expand_dims(log_sums, 1)
    return tf.subtract(log_density,log_sums)
 
def MLE(learning_rate, K, data_to_use=data, num_pts_to_use=num_pts, D=dim, epochs=1000, validation=False, loss_data=False, percentages=False, showProgress=True):
    
    tf.set_random_seed(421)
    #Data
    x = tf.placeholder(tf.float32, shape=(None, D), name='x')
    
    #Variables
    sigma_unbounded = tf.Variable(tf.random_normal([K,1]), name='sigma_unbounded')
    MU = tf.Variable(tf.random_normal([K, D]), name = 'MU')
    proba_pi_unbounded = tf.Variable(tf.random_normal([K,1]), name='proba_pi')
    
    #transforming pi and sigma to bound them with their correct respective constraints
    log_pi = hlp.logsoftmax(proba_pi_unbounded) 
    sigma = tf.exp(sigma_unbounded, name='sigma')
    
    #Calculating loss
    log_PDF = log_GaussPDF(x, MU, sigma)      
    log_weighted_PDF = tf.transpose(log_pi)+log_PDF
    loss_op=-tf.reduce_sum(hlp.reduce_logsumexp(log_weighted_PDF))

    #Determing best cluster for each point
    log_posterior_tensor = log_posterior(log_PDF, log_pi)
    sets = tf.argmax(log_posterior_tensor, 1)
    
    
    ##Optimize Loss
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.99, epsilon=1e-5)
    train_op = optimizer.minimize(loss_op)
         
    init = tf.global_variables_initializer()
    trainLossHistory = np.zeros(epochs)
    
    trainData=data_to_use
    # For Validation set
    if validation:
      valid_batch = int(num_pts_to_use / 3.0)
      np.random.seed(45689)
      rnd_idx = np.arange(num_pts_to_use)
      np.random.shuffle(rnd_idx)
      valdata = data_to_use[rnd_idx[:valid_batch]]
      trainData = data_to_use[rnd_idx[valid_batch:]] 

    with tf.Session() as sess:
        sess.run(init)
        print("Starting MLE algorith with K="+str(K))
        for step in range(0, epochs):      
            _, trainLoss, sets_after, mu_out, sigma_out, l_pi= sess.run([train_op, loss_op, sets, MU, sigma, log_pi], feed_dict={x: trainData})
            if(loss_data):
                trainLossHistory[step]=trainLoss
            
            if(step%100==0 and showProgress):
                print("Step "+str(step))
                print(trainLoss)
                plt.scatter(trainData[:, 0], trainData[:, 1], c=sets_after, s=10, alpha=0.1)
                plt.plot(mu_out[:, 0], mu_out[:, 1], 'r+', markersize=20, mew=3)
                plt.show()
        print("Optimization Finished!")
        
        sets_final, mu_final, sigma_final, log_pi_final = sess.run([sets, MU, sigma, log_pi], feed_dict={x: data_to_use})
        pi_final = np.exp(log_pi_final)
        
        validationLoss=0
        if validation:
            validationLoss=sess.run([loss_op], feed_dict={x:valdata})
        
        dict_percentages=0
        if percentages:
            unique, counts = np.unique(sets_final, return_counts=True)
            dict_percentages=counts/np.sum(counts)
            
    return mu_final, trainLossHistory, validationLoss, dict_percentages, sets_final, sigma_final, pi_final 


def getDataExercise21():
    print("Getting data for exercise 2.1...")
    mu_final, trainLossHistory, _, _,sets_final, sigma_final, pi_final = MLE(1e-2, K=3, epochs=1200, loss_data=True)
    with open('exercise21.pkl', 'wb') as f: 
        pickle.dump([mu_final, trainLossHistory, sets_final, sigma_final, pi_final], f)
    return

def getDataExercise22():
    print("Getting data for exercise 2.2...")
    mu_final1, _, validationLoss1, _, sets_final1, sigma_final1, pi_final1 = MLE(1e-2, K=1, validation=True)
    mu_final2, _, validationLoss2, _, sets_final2, sigma_final2, pi_final2 = MLE(1e-2, K=2, validation=True)
    mu_final3, _, validationLoss3, _, sets_final3, sigma_final3, pi_final3 = MLE(1e-2, K=3, validation=True, epochs = 5000)
    mu_final4, _, validationLoss4, _, sets_final4, sigma_final4, pi_final4 = MLE(1e-2, K=4, validation=True, epochs = 5000)
    mu_final5, _, validationLoss5, _, sets_final5, sigma_final5, pi_final5 = MLE(1e-2, K=5, validation=True, epochs = 5000)
    with open('exercise22.pkl', 'wb') as f: 
        pickle.dump([mu_final1, validationLoss1, sets_final1, sigma_final1, pi_final1,
                     mu_final2, validationLoss2, sets_final2, sigma_final2, pi_final2,
                     mu_final3, validationLoss3, sets_final3, sigma_final3, pi_final3,
                     mu_final4, validationLoss4, sets_final4, sigma_final4, pi_final4,
                     mu_final5, validationLoss5, sets_final5, sigma_final5, pi_final5
                     ], f)
    return

def getDataExercise23part1():
    print('Getting data for exercise 2.3 part 1...')
    data_new = np.load('data100D.npy')
    [num_pts_new, dim_new] = np.shape(data_new)
    
    mu_final5, trainLossHistory5, validationLoss5, dict_percentages5, sets_final5, sigma_final5, pi_final5 = MLE(1e-2, data_to_use=data_new, num_pts_to_use=num_pts_new, K=5, D=dim_new, validation=True, loss_data=True, percentages=True, epochs=3000) 
    mu_final10, trainLossHistory10, validationLoss10, dict_percentages10, sets_final10, sigma_final10, pi_final10 = MLE(1e-2, data_to_use=data_new, num_pts_to_use=num_pts_new, K=10, D=dim_new, validation=True, loss_data=True, percentages=True, epochs=3000) 
    mu_final15, trainLossHistory15, validationLoss15, dict_percentages15, sets_final15, sigma_final15, pi_final15 = MLE(1e-2, data_to_use=data_new, num_pts_to_use=num_pts_new, K=15, D=dim_new, validation=True, loss_data=True, percentages=True, epochs=3000) 
    mu_final20, trainLossHistory20, validationLoss20, dict_percentages20, sets_final20, sigma_final20, pi_final20 = MLE(1e-2, data_to_use=data_new, num_pts_to_use=num_pts_new, K=20, D=dim_new, validation=True, loss_data=True, percentages=True, epochs=3000) 
    mu_final30, trainLossHistory30, validationLoss30, dict_percentages30, sets_final30, sigma_final30, pi_final30 = MLE(1e-2, data_to_use=data_new, num_pts_to_use=num_pts_new, K=30, D=dim_new, validation=True, loss_data=True, percentages=True, epochs=3000) 
    with open('exercise231.pkl', 'wb') as f: 
        pickle.dump([ mu_final5, trainLossHistory5, validationLoss5, dict_percentages5, sets_final5, sigma_final5, pi_final5,
                    mu_final10, trainLossHistory10, validationLoss10, dict_percentages10, sets_final10, sigma_final10, pi_final10,
                    mu_final15, trainLossHistory15, validationLoss15, dict_percentages15, sets_final15, sigma_final15, pi_final15,
                    mu_final20, trainLossHistory20, validationLoss20, dict_percentages20, sets_final20, sigma_final20, pi_final20,
                    mu_final30, trainLossHistory30, validationLoss30, dict_percentages30, sets_final30, sigma_final30, pi_final30], f)
    return
    
def getDataExercise23part2():
    print('Getting data for exercise 2.3 part 2...')
    data_new = np.load('data100D.npy')
    [num_pts_new, dim_new] = np.shape(data_new)
    
    mu_final5, trainLossHistory5, validationLoss5, dict_percentages5, sets_final5 = km.kMeans(1e-2, K=5, data_to_use=data_new, num_pts_to_use=num_pts_new, D=dim_new, epochs=3000, validation=True, loss_data=True, percentages=True) 
    mu_final10, trainLossHistory10, validationLoss10, dict_percentages10, sets_final10 = km.kMeans(1e-2, K=10, data_to_use=data_new, num_pts_to_use=num_pts_new, D=dim_new, epochs=3000, validation=True, loss_data=True, percentages=True) 
    mu_final15, trainLossHistory15, validationLoss15, dict_percentages15, sets_final15 = km.kMeans(1e-2, K=15, data_to_use=data_new, num_pts_to_use=num_pts_new, D=dim_new, epochs=3000, validation=True, loss_data=True, percentages=True) 
    mu_final20, trainLossHistory20, validationLoss20, dict_percentages20, sets_final20 = km.kMeans(1e-2, K=20, data_to_use=data_new, num_pts_to_use=num_pts_new, D=dim_new, epochs=3000, validation=True, loss_data=True, percentages=True) 
    mu_final30, trainLossHistory30, validationLoss30, dict_percentages30, sets_final30 = km.kMeans(1e-2, K=30, data_to_use=data_new, num_pts_to_use=num_pts_new, D=dim_new, epochs=3000, validation=True, loss_data=True, percentages=True) 
    with open('exercise232.pkl', 'wb') as f: 
        pickle.dump([ mu_final5, trainLossHistory5, validationLoss5, dict_percentages5, sets_final5,
                    mu_final10, trainLossHistory10, validationLoss10, dict_percentages10, sets_final10,
                    mu_final15, trainLossHistory15, validationLoss15, dict_percentages15, sets_final15,
                    mu_final20, trainLossHistory20, validationLoss20, dict_percentages20, sets_final20,
                    mu_final30, trainLossHistory30, validationLoss30, dict_percentages30, sets_final30], f)
    return
    
def getDataExercise23():
    getDataExercise23part1()
    getDataExercise23part2()
    return

    
def plotExercise21():
    with open('exercise21.pkl', 'rb') as f:  
        mu_final, trainLossHistory, sets_final, sigma_final, pi_final = pickle.load(f)
        
    startIndex = 0
    endIndex = 1200
    x = range(startIndex, endIndex)
    plt.title("Loss for K=3")
    plt.plot(x,trainLossHistory[startIndex:endIndex], '-b', label='TrainData loss')
    plt.legend(loc='best')
    plt.xlabel('Epochs')
    plt.show()
    print('MU : ')
    print(mu_final)
    print('\nSigma : ')
    print(sigma_final)
    print('\npi : ')
    print(pi_final)   
    return

def plotExercise22():
    with open('exercise22.pkl', 'rb') as f:  
        mu_final1, validationLoss1, sets_final1, sigma_final1, pi_final1, mu_final2, validationLoss2, sets_final2, sigma_final2, pi_final2, mu_final3, validationLoss3, sets_final3, sigma_final3, pi_final3, mu_final4, validationLoss4, sets_final4, sigma_final4, pi_final4, mu_final5, validationLoss5, sets_final5, sigma_final5, pi_final5 = pickle.load(f)
        
        plt.title("Scatter plot for K=1")
        plt.scatter(data[:, 0], data[:, 1], c=sets_final1, s=10, alpha=0.1)
        plt.plot(mu_final1[:, 0], mu_final1[:, 1], 'r+', markersize=20, mew=3)
        plt.show()

        plt.title("Scatter plot for K=2")
        plt.scatter(data[:, 0], data[:, 1], c=sets_final2, s=10, alpha=0.1)
        plt.plot(mu_final2[:, 0], mu_final2[:, 1], 'r+', markersize=20, mew=3)
        plt.show()
        
        plt.title("Scatter plot for K=3")
        plt.scatter(data[:, 0], data[:, 1], c=sets_final3, s=10, alpha=0.1)
        plt.plot(mu_final3[:, 0], mu_final3[:, 1], 'r+', markersize=20, mew=3)
        plt.show()
        
        plt.title("Scatter plot for K=4")
        plt.scatter(data[:, 0], data[:, 1], c=sets_final4, s=10, alpha=0.1)
        plt.plot(mu_final4[:, 0], mu_final4[:, 1], 'r+', markersize=20, mew=3)
        plt.show()     

        plt.title("Scatter plot for K=5")
        plt.scatter(data[:, 0], data[:, 1], c=sets_final5, s=10, alpha=0.1)
        plt.plot(mu_final5[:, 0], mu_final5[:, 1], 'r+', markersize=20, mew=3)
        plt.show()
        
        print('Validation losses:')
        print(validationLoss1)
        print(validationLoss2)
        print(validationLoss3)
        print(validationLoss4)
        print(validationLoss5)
        return     
    
def plotExercise23(want_graphs=True):
    data = np.load('data100D.npy')
    [num_pts, dim] = np.shape(data)
    with open('exercise231.pkl', 'rb') as f:  
        mu_final5, trainLossHistory5, validationLoss5, dict_percentages5, sets_final5, sigma_final5, pi_final5, mu_final10, trainLossHistory10, validationLoss10, dict_percentages10, sets_final10, sigma_final10, pi_final10, mu_final15, trainLossHistory15, validationLoss15, dict_percentages15, sets_final15, sigma_final15, pi_final15, mu_final20, trainLossHistory20, validationLoss20, dict_percentages20, sets_final20, sigma_final20, pi_final20, mu_final30, trainLossHistory30, validationLoss30, dict_percentages30, sets_final30, sigma_final30, pi_final30 = pickle.load(f)
    with open('exercise232.pkl', 'rb') as f:  
        kmu_final5, ktrainLossHistory5, kvalidationLoss5, kdict_percentages5, ksets_final5, kmu_final10, ktrainLossHistory10, kvalidationLoss10, kdict_percentages10, ksets_final10, kmu_final15, ktrainLossHistory15, kvalidationLoss15, kdict_percentages15, ksets_final15, kmu_final20, ktrainLossHistory20, kvalidationLoss20, kdict_percentages20, ksets_final20, kmu_final30, ktrainLossHistory30, kvalidationLoss30, kdict_percentages30, ksets_final30 = pickle.load(f)
    print('Validation losses MoG:')
    print(validationLoss5)
    print(validationLoss10)
    print(validationLoss15)
    print(validationLoss20)
    print(validationLoss30)
    
    print('\Validation losses kMeans:')
    print(kvalidationLoss5)
    print(kvalidationLoss10)
    print(kvalidationLoss15)
    print(kvalidationLoss20)
    print(kvalidationLoss30)
    
    if want_graphs:
        num_graphs=5
        for i in range(0,num_graphs):
            x=np.random.randint(0, 100)
            y=np.random.randint(0, 100)
            plt.title("Scatter plot for MoG K=5")
            plt.scatter(data[:, x], data[:, y], c=sets_final5, s=10, alpha=0.1)
            plt.plot(mu_final5[:, x], mu_final5[:, y], 'r+', markersize=20, mew=3)
            plt.show()
        for i in range(0,num_graphs):
            x=np.random.randint(0, 100)
            y=np.random.randint(0, 100)    
            plt.title("Scatter plot for MoG K=10")
            plt.scatter(data[:, x], data[:, y], c=sets_final10, s=10, alpha=0.1)
            plt.plot(mu_final10[:, x], mu_final10[:, y], 'r+', markersize=20, mew=3)
            plt.show()
        for i in range(0,num_graphs):
            x=np.random.randint(0, 100)
            y=np.random.randint(0, 100)        
            plt.title("Scatter plot for MoG K=15")
            plt.scatter(data[:, x], data[:, y], c=sets_final15, s=10, alpha=0.1)
            plt.plot(mu_final15[:, x], mu_final15[:, y], 'r+', markersize=20, mew=3)
            plt.show()
        for i in range(0,num_graphs):
            x=np.random.randint(0, 100)
            y=np.random.randint(0, 100)        
            plt.title("Scatter plot for MoG K=20")
            plt.scatter(data[:, x], data[:, y], c=sets_final20, s=10, alpha=0.1)
            plt.plot(mu_final20[:, x], mu_final20[:, y], 'r+', markersize=20, mew=3)
            plt.show()     
        for i in range(0,num_graphs):
            x=np.random.randint(0, 100)
            y=np.random.randint(0, 100)    
            plt.title("Scatter plot for MoG K=30")
            plt.scatter(data[:, x], data[:, y], c=sets_final30, s=10, alpha=0.1)
            plt.plot(mu_final30[:, x], mu_final30[:, y], 'r+', markersize=20, mew=3)
            plt.show()
            
            
        for i in range(0,num_graphs):
            x=np.random.randint(0, 100)
            y=np.random.randint(0, 100)
            plt.title("Scatter plot for kMeans K=5")
            plt.scatter(data[:, x], data[:, y], c=ksets_final5, s=10, alpha=0.1)
            plt.plot(kmu_final5[:, x], kmu_final5[:, y], 'r+', markersize=20, mew=3)
            plt.show()
        for i in range(0,num_graphs):
            x=np.random.randint(0, 100)
            y=np.random.randint(0, 100)    
            plt.title("Scatter plot for kMeans K=10")
            plt.scatter(data[:, x], data[:, y], c=ksets_final10, s=10, alpha=0.1)
            plt.plot(kmu_final10[:, x], kmu_final10[:, y], 'r+', markersize=20, mew=3)
            plt.show()
        for i in range(0,num_graphs):
            x=np.random.randint(0, 100)
            y=np.random.randint(0, 100)        
            plt.title("Scatter plot for kMeans K=15")
            plt.scatter(data[:, x], data[:, y], c=ksets_final15, s=10, alpha=0.1)
            plt.plot(kmu_final15[:, x], kmu_final15[:, y], 'r+', markersize=20, mew=3)
            plt.show()
        for i in range(0,num_graphs):
            x=np.random.randint(0, 100)
            y=np.random.randint(0, 100)        
            plt.title("Scatter plot for kMeans K=20")
            plt.scatter(data[:, x], data[:, y], c=ksets_final20, s=10, alpha=0.1)
            plt.plot(kmu_final20[:, x], kmu_final20[:, y], 'r+', markersize=20, mew=3)
            plt.show()     
        for i in range(0,num_graphs):
            x=np.random.randint(0, 100)
            y=np.random.randint(0, 100)    
            plt.title("Scatter plot for kMeans K=30")
            plt.scatter(data[:, x], data[:, y], c=ksets_final30, s=10, alpha=0.1)
            plt.plot(kmu_final30[:, x], kmu_final30[:, y], 'r+', markersize=20, mew=3)
            plt.show()

    
    return

#getDataExercise21()
#getDataExercise22()
#getDataExercise23()
    
#plotExercise21()
#plotExercise22()
#plotExercise23(want_graphs=False)