import scipy.io as sio
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import random
import time
import itertools
import os,psutil

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score #For recognition accuracy
from numpy.linalg import matrix_rank
from numpy.linalg import inv


mat_content = sio.loadmat('face.mat')
face_data = mat_content['X']
face_labels = mat_content['l']

def partition_data(fileIn, fileOut, n):
	#partition ratio is betw 0-1, accepts only increments of 0.1
	mat_content = sio.loadmat(fileIn)
	X = mat_content['X']
	y = mat_content['l']
	
	N = X.shape[1]
	D = X.shape[0]
	#from each class want to take random sample of faces

	y_test = np.array([]).reshape((1,0))
	X_test = np.array([]).reshape((D,0))

	for c in range(1,53):
		cluster = np.where(y == c)[1]
		idx = np.random.choice(cluster, n, replace=False)
		y_test = np.append(y_test, y[:,idx], axis = 1)
		X_test = np.append(X_test, X[:,idx], axis = 1)
		
		X = np.delete(X,idx,1)
		y = np.delete(y,idx,1)

	X_train = X
	y_train = y
		
	matdata = {
		"X_train": X_train,
		"y_train": y_train,
		"X_test": X_test,
		"y_test": y_test,
	}
	sio.savemat(fileOut, matdata)	

def partition_data_old(data, data_labels, partition_ratio):
	N = data.shape[1]
	D = data.shape[0]
	training_N = int(N*(partition_ratio))
	test_N = N - training_N
	test_idx = random.sample(range(N), test_N)
	test_data = np.zeros((D, test_N))
	test_label = np.zeros((1, test_N))

	for i,j in enumerate(test_idx):
		test_data[:,i] = data[:,j]
		test_label[0,i] = data_labels[0,j]

	training_data = np.delete(data,test_idx,1)
	training_label = np.delete(data_labels,test_idx,1)
	test_label = test_label.astype(int)
	
	return training_data, training_label, test_data, test_label
	
def print_img(img):
    img = np.reshape(img,(46,56))
    img = img.T
    plt.imshow(img, cmap = 'gist_gray')
    plt.show()


def NN(vec,mat):
        identity = 0
        e = 0
        min = 0

        for i in range(mat.shape[1]):
                e = np.linalg.norm(vec-mat[:,i])
                if e < min or i==1: 
                        min = e
                        identity = i
        return min, identity

def alternative(training_data, testing_data):
        d=0
        cluster = np.empty((training_data.shape[0],1))
        eigenfaces_alt = np.zeros((testing_data.shape[0],testing_data.shape[1]))
        data = np.zeros((testing_data.shape[0],1))
        error_alt = np.empty((1,testing_data.shape[1]))
        phi_alt = np.zeros(testing_data.shape)
        x = np.zeros((1,testing_data.shape[1]))
        testing_N = testing_data.shape[1]
        training_N = training_data.shape[1]

        for i in range(1,53):
                label = i
                for j in range(training_N):
                        d = training_labels[0,j]
                        if  d == label:
                                data = training_data[:,j]
                                cluster = np.column_stack((cluster,data))
                cluster = np.delete(cluster,0,axis=1)
                eigvals_cluster, eigvecs_cluster=low_dim_PCA(cluster, 10)
                avg_cluster = np.average(cluster, 1)
                for u in range(testing_N):
                        phi_alt[:,u] = testing_data[:,u] - avg_cluster
                omega_alt = np.matmul(phi_alt.T,eigvecs_cluster)
                P = np.matmul(omega_alt,eigvecs_cluster.T)
                for u in range(testing_N):
                        eigenfaces_alt[:,u] = avg_cluster + P.T[:,u]
                for u in range(testing_N):
                        x[0,u] = np.linalg.norm(testing_data[:,u]-eigenfaces_alt[:,u]) #reconstruction error
                error_alt = np.append(error_alt,x,axis=0)
                cluster = np.empty((training_data.shape[0],1))
                x = np.zeros((1,testing_data.shape[1]))

        error_alt = np.delete(error_alt,0,axis=0)

        return error_alt

def PCA(data,M):
        avg = np.average(data, 1)

        phi = np.zeros(data.shape)
        for i in range(data.shape[1]):
                phi[:,i] = data[:,i] - avg

        S = (1/data.shape[1])*np.matmul(phi,phi.T) #Covariance matrix
        eigvals, eigvecs = np.linalg.eig(S)
        #Sorting the eigenvalues and eigenvectors
        idx = eigvals.argsort()[::-1] 
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:,idx]


        for i in range(eigvecs.shape[1]):
                eigvecs[:,i] = eigvecs[:,i]/np.linalg.norm(eigvecs[:,i])

        best_eigvecs = eigvecs[:,:M]
        best_eigvals = eigvals[1:M]

        return best_eigvals, best_eigvecs

def low_dim_PCA(data, M):
        avg = np.average(data, 1)
        phi = np.zeros(data.shape)
        for i in range(data.shape[1]):
                phi[:,i] = data[:,i] - avg

        S = (1/data.shape[1])*np.matmul(phi.T,phi)
        eigvals, eigvecs = np.linalg.eig(S)

        idx = eigvals.argsort()[::-1] 
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:,idx]
        eigvals = eigvals.real #Keeping real part only
        eigvecs = eigvecs.real

        eigvecs = np.matmul(phi,eigvecs) #Why are we doing this??

        for i in range(eigvecs.shape[1]):
                eigvecs[:,i] = eigvecs[:,i]/np.linalg.norm(eigvecs[:,i])

        best_eigvecs = eigvecs[:,:M]
        best_eigvals = eigvals[1:M]

        return best_eigvals, best_eigvecs

def SbSw_calculation(dataset,labels): #data we want to calculate it on, and the corresponding labels
        label = 0
        mi = np.empty((dataset.shape[0],1))
        m = np.average(dataset, 1)
        cluster = np.empty((dataset.shape[0],1))
        d = 0
        valid_labels = np.empty((1,1))

        #Collecting images in clusters
        #Computing class means
        for i in range(1,53):
                label = i
                for j in range(labels.shape[1]):
                        d = labels[0,j]
                        if  d == label:
                                data = dataset[:,j]
                                cluster = np.column_stack((cluster,data))
                cluster = np.delete(cluster,0,axis=1)
                if cluster.shape[1] >= 1:
                        valid_labels = np.append(valid_labels,label)
                        cluster = np.average(cluster,1)
                mi = np.column_stack((mi,cluster))
                cluster = np.empty((dataset.shape[0],1))

        valid_labels = np.delete(valid_labels,0)

        mi = np.delete(mi,0,axis=1)
        mi_norm = np.zeros(mi.shape)

        for i in range(mi_norm.shape[1]):
                mi_norm[:,i] = mi[:,i]-m

        Sb = np.matmul(mi_norm,mi_norm.T) #this should be rank c-1
        Sw = np.zeros(Sb.shape)

        for i in valid_labels:
                label = i
                for j in range(labels.shape[1]):
                        d = labels[0,j]
                        if  d == label:
                                data = dataset[:,j]
                                cluster = np.column_stack((cluster,data))
                cluster = np.delete(cluster,0,axis=1)
                cluster_mean = np.average(cluster,1)
                for p in range(cluster.shape[1]):
                        cluster[:,p] = cluster[:,p] - cluster_mean #Remove mean for that class
                Si = np.matmul(cluster,cluster.T)
                Sw = Sw + Si
                cluster = np.empty((dataset.shape[0],1)) 
        
        #Sw should be rank N-c
        
        return Sb, Sw

def LDA(Sw, Sb, Mlda):
        #We compute the generalized eigenvalues and eigenvectors
        F = np.matmul(inv(Sw),Sb)
        gen_eigvals, gen_eigvecs = np.linalg.eig(F)
        gen_eigvals = gen_eigvals.real
        gen_eigvecs = gen_eigvecs.real

        #Sorting the generalized eigenvalues and eigenvectors
        idx = gen_eigvals.argsort()[::-1] 
        gen_eigvals = gen_eigvals[idx]
        gen_eigvecs = gen_eigvecs[:,idx]

        #We pick the largest M eigenvalues
        best_gen_eigvals = gen_eigvals[1:Mlda]
        best_gen_eigvecs = gen_eigvecs[:,:Mlda]

        return best_gen_eigvals, best_gen_eigvecs

def dataEnsemble(training_data, training_labels, testing_data, testing_labels, Mpca, Mlda, c1, T):
	# #We sample the training set omega (PCA data) 
	print("Performing Random Sample Ensemble...")
	print("0%", end='\r')
	c = training_labels[0,-1]

	eigval, wPca = low_dim_PCA(training_data, Mpca)
	avg = np.average(training_data, 1)
	avg = avg[:,np.newaxis]

	phiTrain = training_data - avg
	omegaTrain = np.matmul(wPca.T, phiTrain) 
	phiTest = testing_data - avg
	omegaTest = np.matmul(wPca.T, phiTest)

	ypred = np.zeros((T, testing_data.shape[1]))
	ypredEnsemble = np.zeros((1, testing_data.shape[1]))
	ytrue = testing_labels
	
	errorAvg = 0
	errorCom = 0

	for t in range(T):
		dataSubsample = np.array([]).reshape((omegaTrain.shape[0],0)) #initialise empty
		labelSubsample = np.array([]).reshape((1,0))
		
		classes = np.asarray(random.sample(range(1,c), c1)) #take c1 random selection of classes
		classes = (classes.astype(int)).T #Convert from float to integer
		
		for j in classes: #loop through selected classes, put corresponding labels & data into dataSubsample 
			idx = np.where(training_labels == j)[1]
			dataSubsample = np.append(dataSubsample, omegaTrain[:,idx], axis = 1)
			labelSubsample = np.append(labelSubsample, training_labels[:,idx], axis = 1)
		labelSubsample = labelSubsample.astype(int)
		
		Sb_sample,Sw_sample = SbSw_calculation(dataSubsample, labelSubsample)
		eigvals, wLdaSample = LDA(Sw_sample, Sb_sample, Mlda)
		
		trainProj = np.matmul(wLdaSample.T,dataSubsample)
		testProj = np.matmul(wLdaSample.T,omegaTest) #Project data into LDA feature space    
		
		for i in range(testing_data.shape[1]):
			error, identity = NN(testProj[:,i], trainProj)
			ypred[t,i] = labelSubsample[:,identity]
		
		errorAvg = errorAvg + (1-classification_accuracy(ytrue, ypred[t,:])) 
		
		print((t+1)*100/T,'%', end='\r')

	ypred = ypred.astype(int)	
	for i in range(testing_data.shape[1]):
		counts = np.bincount(ypred[:,i])
		ypredEnsemble[:,i] = np.argmax(counts)

	errorAvg = errorAvg/T
	errorCom = 1 - classification_accuracy(ytrue, ypredEnsemble)

	print("Average Error: ", errorAvg)
	print("Committee Error: ", errorCom)
	
	print("Completed Random Sample Ensemble.\n")
	return ypredEnsemble, ytrue, errorCom, errorAvg

def dataEnsemblePCA(training_data, training_labels, testing_data, testing_labels, Mpca, Mlda, c1, T):
	# #We sample the training set omega (PCA data) 
	print("Performing Random Sample Ensemble...")
	print("0%", end='\r')
	c = training_labels[0,-1]

	ypred = np.zeros((T, testing_data.shape[1]))
	ypredEnsemble = np.zeros((1, testing_data.shape[1]))
	ytrue = testing_labels

	for t in range(T):
		dataSubsample = np.array([]).reshape((training_data.shape[0],0)) #initialise empty
		labelSubsample = np.array([]).reshape((1,0))
		
		classes = np.asarray(random.sample(range(1,c), c1)) #take c1 random selection of classes
		classes = (classes.astype(int)).T #Convert from float to integer
		
		for j in classes: #loop through selected classes, put corresponding labels & data into dataSubsample 
			idx = np.where(training_labels == j)[1]
			dataSubsample = np.append(dataSubsample, training_data[:,idx], axis = 1)
			labelSubsample = np.append(labelSubsample, training_labels[:,idx], axis = 1)
		labelSubsample = labelSubsample.astype(int)
		
		eigval, wPca = low_dim_PCA(dataSubsample, Mpca)
		avg = np.average(dataSubsample, 1)
		avg = avg[:,np.newaxis]

		phiTrain = dataSubsample - avg
		omegaTrain = np.matmul(wPca.T, phiTrain) 
		phiTest = testing_data - avg
		omegaTest = np.matmul(wPca.T, phiTest)
	
		
		Sb_sample,Sw_sample = SbSw_calculation(dataSubsample, labelSubsample)
		eigvals, wLdaSample = LDA(Sw_sample, Sb_sample, Mlda)
		
		trainProj = np.matmul(wLdaSample.T,dataSubsample)
		testProj = np.matmul(wLdaSample.T,omegaTest) #Project data into LDA feature space    
		
		for i in range(testing_N):
			error, identity = NN(testProj[:,i], trainProj)
			ypred[t,i] = labelSubsample[:,identity]
		
		print((t+1)*100/T,'%', end='\r')

	ypred = ypred.astype(int)	
	for i in range(testing_data.shape[1]):
		counts = np.bincount(ypred[:,i])
		ypredEnsemble[:,i] = np.argmax(counts)

	print("Completed Random Sample Ensemble.")
	return ypredEnsemble, ytrue
	
def featureEnsemble(training_data, training_labels, testing_data, testing_labels, Mpca, Mlda, M0, M1, T):
	print("Performing Random Feature Ensemble...")
	print("0%", end='\r')
	eigval, wPca = low_dim_PCA(training_data, Mpca)
	ypred = np.zeros((T,testing_data.shape[1]))
	ypredEnsemble = np.zeros((1,testing_data.shape[1]))
	ytrue = testing_labels

	avg = np.average(training_data, 1)
	avg = avg[:,np.newaxis]
	phi = training_data - avg
	phiTest = testing_data - avg
	
	errorAvg = 0
	errorCom = 0

	for t in range(T):
		wSubSpace1 = wPca[:,0:M0] # shape (2576, M0)
		r = np.asarray(random.sample(range(M0,Mpca), M1))
		r = (r.astype(int)).T #Convert from float to integer
		wSubSpace2 = wPca[:,r]
		wSubSpace = np.concatenate((wSubSpace1,wSubSpace2), axis = 1)

		#project data onto wPca
		omegaTrain = np.matmul(wSubSpace.T,phi) 

		Sb, Sw = SbSw_calculation(omegaTrain, training_labels)
		eigval, wLda = LDA(Sw, Sb, Mlda) #shape (M0+M1,25)

		#project lowdim data onto wLda
		projection = np.matmul(wLda.T,omegaTrain)

		testProj = np.matmul(wSubSpace.T,phiTest)
		testProj = np.matmul(wLda.T,testProj)

		identity = np.zeros((1,testing_data.shape[1]))

		for i in range(testing_data.shape[1]):
			error, identity = NN(testProj[:,i], projection)
			ypred[t,i] = training_labels[:,identity]
			
		errorAvg = errorAvg + (1-classification_accuracy(ytrue, ypred[t,:])) 

		print((t+1)*100/T,'%', end='\r')
		
	ypred = ypred.astype(int)	
	for i in range(testing_data.shape[1]):
		counts = np.bincount(ypred[:,i])
		ypredEnsemble[:,i] = np.argmax(counts)
		
	errorAvg = errorAvg/T
	errorCom = 1 - classification_accuracy(ytrue, ypredEnsemble)
	
	print("Average Error: ", errorAvg)
	print("Committee Error: ", errorCom)
	
	print("Completed Random Feature Ensemble.\n")
	return ypredEnsemble, ytrue, errorCom, errorAvg

def plot_confusion_matrix(cm,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
	#https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    #else:
        #print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

def classification_accuracy(ytrue, ypred):
#?????
	compare = np.equal(ytrue, ypred)
	correct = np.sum(compare)
	accuracy = correct/ytrue.shape[1]
	
	return accuracy

partition_data('face.mat', 'face_partition.mat', n=3)
	
mat_content = sio.loadmat('face_partition.mat')
training_data = mat_content['X_train']
testing_data = mat_content['X_test']
training_labels = mat_content['y_train']
testing_labels = mat_content['y_test']
	
#--------------------------------------------------------------------------------------------------------------------- PCA
print(time.clock())

eigvals, eigvecs = PCA(training_data,200)
nonzero = np.count_nonzero(eigvals)
print(nonzero)

best_eigvals = eigvals[np.where(eigvals > 9000)] 
print(best_eigvals)

#--------------------------------------------------------------------------------------------- Low dimensional computation
best_eigvals_lowdim, best_eigvecs_lowdim = low_dim_PCA(training_data,200)

print(time.clock())
plt.plot(best_eigvals_lowdim)
plt.show()

#-------------------------------------------------------------------------------------------------------------- Eigenfaces
best_eigvals_lowdim, best_eigvecs_lowdim = low_dim_PCA(training_data,200)
avg = np.average(training_data, 1)
phi = np.zeros(training_data.shape)
best_eigvals, best_eigvecs = low_dim_PCA(training_data,200)

for i in range(training_N):
        phi[:,i] = training_data[:,i] - avg
a = np.matmul(phi.T, best_eigvecs_lowdim) #Omega
P = np.matmul(a,best_eigvecs_lowdim.T) 
P = P.T
eigfaces = np.zeros(P.shape) 

for i in range(P.shape[1]):
        eigfaces[:,i]=avg+P[:,i]

eigface_5 = avg + P.T[:,5]
eigface_5 = eigface_5.astype(int)
print_img(eigface_5)
print_img(training_data[:,5])
#---------------------------------------------------------------------------------------------------------------- Testing
omega_n = a.T
phi_test = np.zeros(testing_data.shape)

for i in range(testing_N):
    phi_test[:,i] = testing_data[:,i] - avg

omega = np.matmul(phi_test.T,best_eigvecs_lowdim)
omega = omega.T

#----------------------------------------------------------------------------------------------- Nearest Neighbour method
error_vec = np.zeros((1,testing_N))
identity_vec = np.zeros((1,testing_N))
predicted_labels = np.zeros((1,testing_N))

for i in range(testing_N):
        error_vec[0,i], identity_vec[0,i] = NN(omega[:,i],omega_n)

identity_vec = identity_vec.astype(int)

for i in range(testing_N):
        p = identity_vec[0,i]
        predicted_labels[0,i] = training_labels[0,p]

#print(confusion_matrix(testing_labels.T, predicted_labels.T))

#----------------------------------------------------------------------------------------------------- Alternative method
erreur = alternative(training_data,testing_data)

#Testing out the alternative method
p = 0
labela = np.argmin(erreur[:,20]) #image 10 testing data label
print_img(testing_data[:,20])


for i in range(training_N):
        if training_labels[0,i] == labela+1:
                p = i
                break
print_img(training_data[:,p])

#---------------------------------------------------------------------------------------------------------- PCA-LDA method
pca_eigvals, wpca = low_dim_PCA(training_data,313) 

omega_pca = np.matmul(phi.T, wpca) #Datapoints in PCA feature space
omega_pca = omega_pca.T

Sb, Sw = SbSw_calculation(omega_pca,training_labels)

lda_eigvals, wlda = LDA(Sw, Sb, 25)  #Applying LDA

#DOES THE PCA DATA NEED TO BE NORMALISED??
train_proj = np.matmul(wlda.T,omega_pca) #Classification space projection for training data

for i in range(P_fish.shape[1]):
        fisherfaces[:,i]=avg+P_fish[:,i]

a = np.matmul(phi.T, best_eigvecs) #Omega
P = np.matmul(a,best_eigvecs_lowdim.T) 
P = P.T
eigfaces = np.zeros(P.shape) 

for i in range(P.shape[1]):
        eigfaces[:,i]=avg+P[:,i]

#--------------------------------------------------------------------------------------------------------- Nearest Neighbour for LDA-PCA
omega_test = np.matmul(phi_test.T,wpca)
omega_test = omega_test.T

error_lda = np.zeros((1,testing_N))
identity_lda = np.zeros((1,testing_N))
predicted_labels_lda = np.zeros((1,testing_N))

test_proj = np.matmul(wlda.T,omega_test) #Project that data to feature space

for i in range(testing_N):
        error_lda[0,i], identity_lda[0,i] = NN(test_proj[:,i],train_proj)

identity_lda = identity_lda.astype(int)

for i in range(testing_N):
        p = identity_lda[0,i]
        predicted_labels_lda[0,i] = training_labels[0,p]

"""
o = identity_lda[0,6]
print_img(training_data[:,o])

i = 0
for i in range(testing_N):
        l = training_labels[0,i]
        if l == training_labels[0,o]:
                break
        else: 
                i = i+1

print_img(training_data[:,i])
"""

#PLOTS OF EIGENFACES
eigvecs = 255 * (1.0 - eigvecs)
eigvecs = eigvecs.astype(int)

fig, axes = plt.subplots(5, 10, figsize=(7, 6), sharex=True, sharey=True)
ax = axes.ravel()

for i in range(50):
        img = (np.reshape(eigvecs[:,200-i],(46,56))).T
        ax[i].imshow(img, cmap='gist_gray')

plt.show()

#RECONSTRUCTION OF A FEW IMAGES
recons_error = np.zeros(1,training_data.shape[1])
eigvals1, eigvecs1 = low_dim_PCA(training_data,200)
avg = np.average(training_data, 1)
phi = np.zeros(training_data.shape)

for i in range(training_data.shape[1]):
    phi[:,i] = training_data[:,i] - avg

a = np.matmul(phi.T, eigvecs1) #Omega
P = np.matmul(a,eigvecs1.T) 
P = P.T

recons = avg + P[:,5]
recons = recons.astype(int)
#print_img(recons)
#print_img(training_data[:,5])

#RECONSTRUCTION OF ALL THE IMAGES TO DETERMINE RECONSTRUCTION ERROR WHEN WE VARY M

avg = np.average(training_data, 1)
recons_error = np.zeros((364,training_data.shape[1]))

phi = np.zeros(training_data.shape)
for i in range(training_data.shape[1]):
    phi[:,i] = training_data[:,i] - avg
recons = np.zeros(training_data.shape)

for i in range(364):
    eigvals, eigvecs = low_dim_PCA(training_data,i)
    a = np.matmul(phi.T, eigvecs) #Omega
    P = np.matmul(a,eigvecs.T) 
    P = P.T
    for j in range(training_data.shape[1]):
        recons = avg + P[:,j]
        recons_error[i,j] = np.linalg.norm(training_data[:,j]-recons)

recons_error = np.average(recons_error, 1)
plt.plot(recons_error)
plt.show()

#NN CLASSIFICATION

pid = os.getpid() 
ps = psutil.Process(pid)

time_start = time.clock() 

avg = np.average(training_data, 1)
phi = np.zeros(training_data.shape)
for i in range(training_data.shape[1]):
    phi[:,i] = training_data[:,i] - avg
phi_test = np.zeros(testing_data.shape)

eigvals, eigvecs = low_dim_PCA(training_data,10)

omega_train = (np.matmul(phi.T, eigvecs)).T

for i in range(testing_data.shape[1]):
    phi_test[:,i] = testing_data[:,i] - avg

omega_test = (np.matmul(phi_test.T,eigvecs)).T

error_vec = np.zeros((1,testing_data.shape[1]))
identity_vec = np.zeros((1,testing_data.shape[1]))
predicted_labels = np.zeros((1,testing_data.shape[1]))

for i in range(testing_data.shape[1]):
        error_vec[0,i], identity_vec[0,i] = NN(omega_test[:,i],omega_train)

identity_vec = identity_vec.astype(int)

for i in range(testing_data.shape[1]):
        p = identity_vec[0,i]
        predicted_labels[0,i] = training_labels[0,p]

memoryUse = ps.memory_info()
print(memoryUse)

time_elapsed = (time.clock() - time_start)
print(time_elapsed)

print(classification_accuracy(testing_labels,predicted_labels))

cm = confusion_matrix(testing_labels.T, predicted_labels.T)
plot_confusion_matrix(cm)
plt.show()

#Success/Failure case
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')

#Class 1
idx = np.where(training_labels == 1)[1]
c1 = omega_train[:,idx]
X1 = c1[0,:] 
Y1 = c1[1,:]
Z1 = c1[2,:] 
ax.scatter(X1,Y1,Z1,c='b',marker='o',label='Class 1') 

#Class 24
idx = np.where(training_labels == 24)[1]
c7 = omega_train[:,idx]
X7 = c7[0,:] 
Y7 = c7[1,:]
Z7 = c7[2,:] 
ax.scatter(X7,Y7,Z7,c='r',marker='o',label='Class 24') 

#Test example - predicted class 1, actual class 7
T1 = omega_test[0,0] 
T2 = omega_test[1,0] 
T3 = omega_test[2,0]  
ax.scatter(T1,T2,T3,c='k',marker='*',label='Failure: Predicted 24, Actual 1') 

T1 = omega_test[0,2] 
T2 = omega_test[1,2] 
T3 = omega_test[2,2]
ax.scatter(T1,T2,T3,c='k',marker='x',label='Success: Predicted 1, Actual 1') 

ax.set_xlabel('u1')
ax.set_ylabel('u2')
ax.set_zlabel('u3')
ax.legend()
plt.show()

plt.title('Class 1 Image misclassified as Class 24')
print_img(testing_data[:,0])
plt.title('Nearest match (Class 1)')
print_img(training_data[:, identity_vec[0,0]])
plt.title('Class 1 Image correctly classified')
print_img(testing_data[:,2])
plt.title('Nearest match (Class 1)')
print_img(training_data[:, identity_vec[0,2]])


#-------ALTERNATIVE METHOD

pid = os.getpid() 
ps = psutil.Process(pid)

time_start = time.clock() 

error_alt = alternative(training_data,testing_data)
predicted_labels = np.argmin(error_alt, axis=0) + 1

print(np.where(training_labels == 18)[1])
print(np.where(training_labels == 2)[1])

fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')

#Class 2
idx = np.where(training_labels == 2)[1]
c1 = omega_train[:,idx]
X1 = c1[0,:] 
Y1 = c1[1,:]
Z1 = c1[2,:] 
ax.scatter(X1,Y1,Z1,c='b',marker='o',label='Class 2') 

#Class 3
idx = np.where(training_labels == 16)[1]
c7 = omega_train[:,idx]
X7 = c7[0,:] 
Y7 = c7[1,:]
Z7 = c7[2,:] 
ax.scatter(X7,Y7,Z7,c='r',marker='o',label='Class 16') 

#Test example - predicted class 1, actual class 7
T1 = omega_test[0,52] 
T2 = omega_test[1,52] 
T3 = omega_test[2,52]  
ax.scatter(T1,T2,T3,c='k',marker='*',label='Failure: Predicted 2, Actual 16') 

T1 = omega_test[0,3] 
T2 = omega_test[1,3] 
T3 = omega_test[2,3]
ax.scatter(T1,T2,T3,c='k',marker='x',label='Success: Predicted 2, Actual 16') 

ax.set_xlabel('u1')
ax.set_ylabel('u2')
ax.set_zlabel('u3')
ax.legend()
plt.show()

r = np.average(training_data[:, 105:111],1)
p = np.average(training_data[:, 7:13],1)

plt.title('Class 16 Image misclassified as Class 2')
print_img(testing_data[:,45])
plt.title('Average Class 16 Image')
print_img(r)
plt.title('Class 2 Image correctly classified')
print_img(testing_data[:,3])
plt.title('Average Class 2 Image')
print_img(p)

predicted_labels = predicted_labels[:,np.newaxis]
idx = np.where(predicted_labels == 2)
print(idx)

time_elapsed = (time.clock() - time_start)
print(time_elapsed)

print(predicted_labels)
print(testing_labels)

memoryUse = ps.memory_info()
print(memoryUse)

print(classification_accuracy(testing_labels,predicted_labels))

cm = confusion_matrix(testing_labels.T, predicted_labels.T)
plot_confusion_matrix(cm)
plt.show()

