import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import helper as hlp
import pickle

# Loading data
data = np.load('data2D.npy')
#data = np.load('data100D.npy')
[num_pts, dim] = np.shape(data)


# Distance function for K-means
def distanceFunc(X, MU):
    # Inputs
    # X: is an NxD matrix (N observations and D dimensions)
    # MU: is an KxD matrix (K means and D dimensions)
    # Outputs
    # pair_dist: is the pairwise distance matrix (NxK)
    expand_x = tf.expand_dims(X, 1)
    expand_mu = tf.expand_dims(MU, 0)
    pair_dist = tf.reduce_sum(tf.square(tf.subtract(expand_x, expand_mu)), 2)
    return pair_dist
    
def kMeans(learning_rate, K, data_to_use=data, num_pts_to_use=num_pts, D=dim, epochs=1000, validation=False, loss_data=False, percentages=False, showProgress=True):
    tf.set_random_seed(421)
    MU = tf.Variable(tf.random_normal([K, D]), name = 'MU')
    x = tf.placeholder(tf.float32, shape=(None, D), name='x')
    expand_x = tf.expand_dims(x, 0)
    expand_mu = tf.expand_dims(MU, 1)
    
    ##CONSTRUCT SETS
    distances = tf.reduce_sum(tf.square(tf.subtract(expand_x, expand_mu)), 2)
    sets = tf.argmin(distances, 0)
    
    ##Calculate Loss
    loss_op=tf.reduce_sum(tf.reduce_min(distances, axis=0))
    
    ##Optimize Loss
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.99, epsilon=1e-5)
    train_op = optimizer.minimize(loss_op)
         
    init = tf.global_variables_initializer()
    trainLossHistory = np.zeros(epochs)
    
    trainData=data
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
        print("Starting K-Means algorith with K="+str(K))
        for step in range(0, epochs):      
            _, trainLoss, sets_after, mu_out= sess.run([train_op, loss_op, sets, MU], feed_dict={x: trainData})
            if(loss_data):
                trainLossHistory[step]=trainLoss
            
            if(step%100==0 and showProgress):
                print("Step "+str(step))
                print(trainLoss)
                plt.scatter(trainData[:, 0], trainData[:, 1], c=sets_after, s=10, alpha=0.1)
                plt.plot(mu_out[:, 0], mu_out[:, 1], 'r+', markersize=20, mew=3)
                plt.show()
        print("Optimization Finished!")
        
        sets_final, mu_final = sess.run([sets, MU], feed_dict={x: data_to_use})
        
        validationLoss=0
        if validation:
            validationLoss=sess.run([loss_op], feed_dict={x:valdata})
        
        dict_percentages=0
        if percentages:
            unique, counts = np.unique(sets_final, return_counts=True)
            dict_percentages=counts/np.sum(counts)
            
    return mu_final, trainLossHistory, validationLoss, dict_percentages, sets_final 




def getDataExercise11():
    print("Getting data for exercise 1.1...")
    mu_final, trainLossHistory, _, _,sets_final = kMeans(1e-2, K=3, loss_data=True)
    with open('exercise11.pkl', 'wb') as f: 
        pickle.dump([mu_final, trainLossHistory, sets_final], f)
    return

def getDataExercise12():
    print("Getting data for exercise 1.2...")
    mu_final1, _, _, dict_percentages1, sets_final1 = kMeans(1e-2, K=1, percentages=True)
    mu_final2, _, _, dict_percentages2, sets_final2 = kMeans(1e-2, K=2, percentages=True)
    mu_final3, _, _, dict_percentages3, sets_final3 = kMeans(1e-2, K=3, percentages=True)
    mu_final4, _, _, dict_percentages4, sets_final4 = kMeans(1e-2, K=4, percentages=True)
    mu_final5, _, _, dict_percentages5, sets_final5 = kMeans(1e-2, K=5, percentages=True)
    with open('exercise12.pkl', 'wb') as f: 
        pickle.dump([mu_final1, dict_percentages1, sets_final1,
                     mu_final2, dict_percentages2, sets_final2,
                     mu_final3, dict_percentages3, sets_final3,
                     mu_final4, dict_percentages4, sets_final4,
                     mu_final5, dict_percentages5, sets_final5], f)
    return

def getDataExercise13():
    print("Getting data for exercise 1.3...")
    mu_final1, _, validationLoss1, _, sets_final1 = kMeans(1e-2, K=1, validation=True)
    mu_final2, _, validationLoss2, _, sets_final2 = kMeans(1e-2, K=2, validation=True)
    mu_final3, _, validationLoss3, _, sets_final3 = kMeans(1e-2, K=3, validation=True)
    mu_final4, _, validationLoss4, _, sets_final4 = kMeans(1e-2, K=4, validation=True)
    mu_final5, _, validationLoss5, _, sets_final5 = kMeans(1e-2, K=5, validation=True)
    with open('exercise13.pkl', 'wb') as f: 
        pickle.dump([mu_final1, validationLoss1, sets_final1,
                     mu_final2, validationLoss2, sets_final2,
                     mu_final3, validationLoss3, sets_final3,
                     mu_final4, validationLoss4, sets_final4,
                     mu_final5, validationLoss5, sets_final5
                     ], f)
    return

#getDataExercise11()
#getDataExercise12()
#getDataExercise13()

#mu_final, _, _, dict_percentages = kMeans(1e-2, K=3, validation=False, percentages=True, epochs=1000)
#print(dict_percentages)
#plt.title("Percentages of different classes")
#plt.bar(range(0, 3), dict_percentages, color=["b", "g", "r", "c", "m", "y", "k", "w"])
#plt.show()

def plotExercise11():
    with open('exercise11.pkl', 'rb') as f:  
        _, trainLossHistory,_ = pickle.load(f)
        
    startIndex = 0
    endIndex = 1000
    x = range(startIndex, endIndex)
    plt.title("Loss for K=3")
    plt.plot(x,trainLossHistory[startIndex:endIndex], '-b', label='TrainData loss')
    plt.legend(loc='best')
    plt.xlabel('Epochs')
    plt.show()
    return

def plotExercise12():
    with open('exercise12.pkl', 'rb') as f:  
        mu_final1, dict_percentages1,sets_final1, mu_final2, dict_percentages2,sets_final2, mu_final3, dict_percentages3,sets_final3, mu_final4, dict_percentages4, sets_final4, mu_final5, dict_percentages5, sets_final5 = pickle.load(f)
        plt.title("Scatter plot for K=1")
        plt.scatter(data[:, 0], data[:, 1], c=sets_final1, s=10, alpha=0.1)
        plt.plot(mu_final1[:, 0], mu_final1[:, 1], 'r+', markersize=20, mew=3)
        plt.show()
    
        print(dict_percentages1)
        plt.title("Percentages of different classes for K=1")
        plt.bar(range(0, 1), dict_percentages1, color=["b", "g", "r", "c", "m", "y", "k", "w"])
        plt.show()

        plt.title("Scatter plot for K=2")
        plt.scatter(data[:, 0], data[:, 1], c=sets_final2, s=10, alpha=0.1)
        plt.plot(mu_final2[:, 0], mu_final2[:, 1], 'r+', markersize=20, mew=3)
        plt.show()
        print(dict_percentages2)
        plt.title("Percentages of different classes for K=2")
        plt.bar(range(0, 2), dict_percentages2, color=["b", "r"])
        plt.show()
        
        plt.title("Scatter plot for K=3")
        plt.scatter(data[:, 0], data[:, 1], c=sets_final3, s=10, alpha=0.1)
        plt.plot(mu_final3[:, 0], mu_final3[:, 1], 'r+', markersize=20, mew=3)
        plt.show()
        print(dict_percentages3)
        plt.title("Percentages of different classes for K=3")
        plt.bar(range(0, 3), dict_percentages3, color=["b", "g", "r"])
        plt.show()
        
        plt.title("Scatter plot for K=4")
        plt.scatter(data[:, 0], data[:, 1], c=sets_final4, s=10, alpha=0.1)
        plt.plot(mu_final4[:, 0], mu_final4[:, 1], 'r+', markersize=20, mew=3)
        plt.show()
        print(dict_percentages4)
        plt.title("Percentages of different classes for K=4")
        plt.bar(range(0, 4), dict_percentages4, color=["b", "c", "r", "y"])
        plt.show()        

        plt.title("Scatter plot for K=5")
        plt.scatter(data[:, 0], data[:, 1], c=sets_final5, s=10, alpha=0.1)
        plt.plot(mu_final5[:, 0], mu_final5[:, 1], 'r+', markersize=20, mew=3)
        plt.show()
        print(dict_percentages5)
        plt.title("Percentages of different classes for K=5")
        plt.bar(range(0, 5), dict_percentages5, color=["b", "r", "g", "c", "y"])
        plt.show()
        return 
    
def plotExercise13():
    with open('exercise13.pkl', 'rb') as f:  
        _, validationLoss1, _, _, validationLoss2, _,_, validationLoss3, _, _, validationLoss4, _, _, validationLoss5, _ = pickle.load(f)
    print("Validation loss for K=1 is " + str(validationLoss1))
    print("Validation loss for K=2 is " + str(validationLoss2))
    print("Validation loss for K=3 is " + str(validationLoss3))
    print("Validation loss for K=4 is " + str(validationLoss4))
    print("Validation loss for K=5 is " + str(validationLoss5))
    plt.plot(range(1,6), [validationLoss1, validationLoss2, validationLoss3, validationLoss4, validationLoss5])
    
    
#plotExercise11()
#plotExercise12()
#plotExercise13()
    
#kMeans(1e-2, K=3)