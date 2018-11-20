import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import random
import time
import itertools

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score #For recognition accuracy
from sklearn.neighbors import KNeighborsClassifier
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
                eigvals_cluster, eigvecs_cluster=low_dim_PCA(cluster, 20)
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

        plt.plot(eigvals)
        plt.show()

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

        eigvecs = np.matmul(phi,eigvecs)

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

def dataEnsemble(training_data, training_labels, testing_data, testing_labels, Mpca, Mlda, c1, T, fusiontype='majority'):
	# #We sample the training set omega (PCA data) 
	print("Performing Random Sample Ensemble...")
	print("0%", end='\r')
	c = training_labels[0,-1]
	trainN = training_data.shape[1]
	testN = testing_data.shape[1]
	D = training_data.shape[0]

	#------------ project data onto N-1 dim -------------
	eigval, wPca = low_dim_PCA(training_data, trainN-1)
	avg = np.average(training_data, 1)
	avg = avg[:,np.newaxis]
	phiTrain = training_data - avg
	phiTest = testing_data - avg
	omegaTrain = np.matmul(wPca.T, phiTrain) 
	omegaTest = np.matmul(wPca.T, phiTest)

	ypred = np.zeros((T, testN))
	ypredEnsemble = np.zeros((1, testN))
	ytrue = testing_labels
	
	errorAvg = 0
	errorCom = 0
	probClass = np.zeros((T,testN,c))

	for t in range(T):
		#------------ take random selection of classes -------------
		dataSubsample = np.array([]).reshape((trainN-1,0)) #initialise empty
		labelSubsample = np.array([]).reshape((1,0))
		classes = np.asarray(random.sample(range(1,c+1), c1)) #take c1 random selection of classes
		classes = (classes.astype(int)).T #Convert from float to integer
		
		for j in classes: #loop through selected classes, put corresponding labels & data into dataSubsample 
			idx = np.where(training_labels == j)[1]
			dataSubsample = np.append(dataSubsample, omegaTrain[:,idx], axis = 1)
			labelSubsample = np.append(labelSubsample, training_labels[:,idx], axis = 1)
		labelSubsample = labelSubsample.astype(int)
	
		#-------------- project lowdim data onto wLda -------------
		Sb_sample,Sw_sample = SbSw_calculation(dataSubsample, labelSubsample)
		eigvals, wLdaSample = LDA(Sw_sample, Sb_sample, Mlda)
		trainProj = np.matmul(wLdaSample.T,dataSubsample)
		testProj = np.matmul(wLdaSample.T,omegaTest) #Project data into LDA feature space    
		
		#--------------- perform k nearest neighbors --------------
		neigh = KNeighborsClassifier(n_neighbors=5) #kNN with 5 nearest neighbours considered
		neigh.fit(trainProj.T, np.ravel(labelSubsample)) #fit to data
		#probClass[t,:,:] = neigh.predict_proba(testProj.T) #compute class probs; returns shape [n_samples, n_classes]
		ypred[t,:] = neigh.predict(testProj.T) #predict classes; returns shape [n_samples]
		errorSample = 1-neigh.score(testProj.T,np.ravel(ytrue)) #compute error
		
		errorAvg = errorAvg + errorSample		
		
		print(ypred[t,:])
		print((t+1)*100/T,'%', end='\r')

	#--------------- calculate ensemble predictions ---------------
	ypred = ypred.astype(int)	
	ypredEnsemble = fusion(probClass, ypred, type=fusiontype)
	errorAvg = errorAvg/T
	errorCom = 1-classification_accuracy(ytrue, ypredEnsemble)

	print("Average Error: ", errorAvg)
	print("Committee Error: ", errorCom)
	#print("Predicted Labels: ", ypredEnsemble)
	#print("Actual Labels: ", ytrue)
	
	print("Completed Random Sample Ensemble.")
	return ypredEnsemble, ytrue, errorCom, errorAvg

def dataEnsemblePCA(training_data, training_labels, testing_data, testing_labels, Mpca, Mlda, c1, T, fusiontype='majority'):
	# #We sample the training set omega (PCA data) 
	print("Performing Random Sample Ensemble...")
	print("0%", end='\r')
	c = training_labels[0,-1]
	trainN = training_data.shape[1]
	testN = testing_data.shape[1]
	D = training_data.shape[0]

	#------------ project data onto N-1 dim -------------
	eigval, wPca = low_dim_PCA(training_data, trainN-1)
	avg = np.average(training_data, 1)
	avg = avg[:,np.newaxis]
	phiTrain = training_data - avg
	phiTest = testing_data - avg
	omegaTrain = np.matmul(wPca.T, phiTrain) 
	omegaTest = np.matmul(wPca.T, phiTest)
	
	ypred = np.zeros((T, testN))
	ypredEnsemble = np.zeros((1, testN))
	ytrue = testing_labels
	
	errorAvg = 0
	errorCom = 0
	probClass = np.zeros((T,testN,c))

	for t in range(T):
		#------------ take random selection of classes -------------
		dataSubsample = np.array([]).reshape((trainN-1,0)) #initialise empty
		labelSubsample = np.array([]).reshape((1,0))
		classes = np.asarray(random.sample(range(1,c+1), c1)) #take c1 random selection of classes
		classes = (classes.astype(int)).T #Convert from float to integer
		
		for j in classes: #loop through selected classes, put corresponding labels & data into dataSubsample 
			idx = np.where(training_labels == j)[1]
			dataSubsample = np.append(dataSubsample, omegaTrain[:,idx], axis = 1)
			labelSubsample = np.append(labelSubsample, training_labels[:,idx], axis = 1)
		labelSubsample = labelSubsample.astype(int)
		
		#------------- perform PCA on subsampled data -------------
		eigval, wPcaSample = low_dim_PCA(dataSubsample, Mpca)
		avgSample = np.average(dataSubsample, 1)
		avgSample = avgSample[:,np.newaxis]

		phiSample = dataSubsample - avgSample
		omegaSample = np.matmul(wPcaSample.T, phiSample) 
		phiTestSample = omegaTest - avgSample
		omegaTestSample = np.matmul(wPcaSample.T, phiTestSample)
	
		#-------------- project lowdim data onto wLda -------------
		Sb_sample,Sw_sample = SbSw_calculation(omegaSample, labelSubsample)
		eigvals, wLdaSample = LDA(Sw_sample, Sb_sample, Mlda)
		trainProj = np.matmul(wLdaSample.T,omegaSample)
		testProj = np.matmul(wLdaSample.T,omegaTestSample) #Project data onto LDA feature space    
		
		#--------------- perform k nearest neighbors --------------
		neigh = KNeighborsClassifier(n_neighbors=5) #kNN with 5 nearest neighbours considered
		neigh.fit(trainProj.T, np.ravel(labelSubsample)) #fit to data
		#probClass[t,:,:] = neigh.predict_proba(testProj.T) #compute class probs; returns shape [n_samples, n_classes]
		ypred[t,:] = neigh.predict(testProj.T) #predict classes; returns shape [n_samples]
		errorSample = 1-neigh.score(testProj.T,np.ravel(ytrue)) #compute error
		
		errorAvg = errorAvg + errorSample		
		
		#print(ypred[t,:])
		print((t+1)*100/T,'%', end='\r')

	#--------------- calculate ensemble predictions ---------------
	ypred = ypred.astype(int)	
	ypredEnsemble = fusion(probClass, ypred, type=fusiontype)
	print(ypredEnsemble)
	errorAvg = errorAvg/T
	errorCom = 1-classification_accuracy(ytrue, ypredEnsemble)

	print("Average Error: ", errorAvg)
	print("Committee Error: ", errorCom)
	#print("Predicted Labels: ", ypredEnsemble)
	#print("Actual Labels: ", ytrue)
	
	print("Completed Random Sample Ensemble.")
	return ypredEnsemble, ytrue, errorCom, errorAvg
	
def featureEnsemble(training_data, training_labels, testing_data, testing_labels, Mlda, M0, M1, T, fusiontype='majority'):
	print("Performing Random Feature Ensemble...")
	print("0%", end='\r')
	testN = testing_data.shape[1]
	trainN = training_data.shape[1]
	
	Mpca = trainN
	eigval, wPca = low_dim_PCA(training_data, Mpca)
	ypred = np.zeros((T,testN))
	ypredEnsemble = np.zeros((1,testN))
	ytrue = testing_labels

	avg = np.average(training_data, 1)
	avg = avg[:,np.newaxis]
	phi = training_data - avg
	phiTest = testing_data - avg
	
	errorAvg = 0
	errorCom = 0
	probClass = np.zeros((T,testN,52))
	

	for t in range(T):
		#------------- take Subspace of eigenvectors --------------
		wSubSpace1 = wPca[:,0:M0] # shape (2576, M0)
		r = np.asarray(random.sample(range(M0,Mpca), M1))
		r = (r.astype(int)).T #Convert from float to integer
		wSubSpace2 = wPca[:,r]
		wSubSpace = np.concatenate((wSubSpace1,wSubSpace2), axis = 1)

		#--------------- project data onto Subspace ---------------
		omegaTrain = np.matmul(wSubSpace.T,phi) 
		omegaTest = np.matmul(wSubSpace.T,phiTest)

		#-------------- project lowdim data onto wLda -------------
		Sb, Sw = SbSw_calculation(omegaTrain, training_labels)
		eigval, wLda = LDA(Sw, Sb, Mlda) #shape (M0+M1,25)
		trainProj = np.matmul(wLda.T,omegaTrain)
		testProj = np.matmul(wLda.T,omegaTest)

		#--------------- perform k nearest neighbors --------------
		neigh = KNeighborsClassifier(n_neighbors=5) #kNN with 5 nearest neighbours considered
		neigh.fit(trainProj.T, np.ravel(training_labels)) #fit to data
		probClass[t,:,:] = neigh.predict_proba(testProj.T) #compute class probs; returns shape [n_samples, n_classes]
		ypred[t,:] = neigh.predict(testProj.T) #predict classes; returns shape [n_samples]
		errorSample = 1-neigh.score(testProj.T,np.ravel(ytrue)) #compute error
		
		errorAvg = errorAvg + errorSample
		print((t+1)*100/T,'%', end='\r')
		
	#--------------- calculate ensemble predictions ---------------
	ypred = ypred.astype(int)	
	ypredEnsemble = fusion(probClass, ypred, type=fusiontype)
	errorAvg = errorAvg/T
	errorCom = 1-classification_accuracy(ytrue, ypredEnsemble)
	
	print("Average Error: ", errorAvg)
	print("Committee Error: ", errorCom)
	#print("Predicted Labels: ", ypredEnsemble)
	#print("Actual Labels: ", ytrue)
	
	print("Completed Random Feature Ensemble.\n")
	return ypredEnsemble, ytrue, errorCom, errorAvg

def fusion(probClass, ypred, type = 'majority'):
	y = np.zeros((1,ypred.shape[1]))
	
	if (type == 'majority'):
		for i in range(ypred.shape[1]):
			counts = np.bincount(ypred[:,i])
			y[:,i] = np.argmax(counts)
			
	elif (type == 'sum'):
		prob = np.zeros((probClass.shape[1], probClass.shape[2]))
		for i in range(probClass.shape[0]):
			prob = prob + probClass[i,:,:]
		prob = prob / probClass.shape[0]
		# plt.figure()
		# plt.plot(prob[5,:])
		# plt.show()
		y = np.argmax(prob, axis=1)
		y = y+1 #since indexing starts at 0 whereas labels start at 1

	elif (type == 'product'):
		prob = np.ones((probClass.shape[1], probClass.shape[2]))
		for i in range(probClass.shape[0]):
			prob = np.multiply(prob, probClass[i,:,:])

		for j in range(prob.shape[0]):
			if any(prob[j,:]) != 0:
				prob[j,:] = prob[j,:] / np.sum(prob[j,:])
				y[0,j] = np.argmax(prob[j,:])
				y[0,j] = y[0,j]+1

		# plt.figure()
		# plt.plot(prob[20,:])
		# plt.show()
		
	else:
		print("Invalid fusion type")

	return y
			
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

#just use neigh.score
def classification_accuracy(ytrue, ypred):
#?????
	compare = np.equal(ytrue, ypred)
	correct = np.sum(compare)
	accuracy = correct/ytrue.shape[1]
	
	return accuracy

#partition_data('face.mat', 'face_partition.mat', n=3)

# idx = np.where(face_labels == 7)[1]
# for i in range(len(idx)):
	# print_img(face_data[:,idx[i]])
	
mat_content = sio.loadmat('face_partition.mat')
training_data = mat_content['X_train']
testing_data = mat_content['X_test']
training_labels = mat_content['y_train']
testing_labels = mat_content['y_test']
	
#--------------------------------------------------------------------------------------------------------------------- PCA
#print(time.clock())

# eigvals, eigvecs = PCA(training_data,200)
# nonzero = np.count_nonzero(eigvals)
# print(nonzero)

# """best_eigvals = eigvals[np.where(eigvals > 9000)] 
# print(best_eigvals)"""

# #--------------------------------------------------------------------------------------------- Low dimensional computation
# best_eigvals_lowdim, best_eigvecs_lowdim = low_dim_PCA(training_data,200)

# #print(time.clock())
# """plt.plot(best_eigvals_lowdim)
# plt.show()"""

# #-------------------------------------------------------------------------------------------------------------- Eigenfaces
# best_eigvals_lowdim, best_eigvecs_lowdim = low_dim_PCA(training_data,200)
# avg = np.average(training_data, 1)
# phi = np.zeros(training_data.shape)
# best_eigvals, best_eigvecs = low_dim_PCA(training_data,200)

# for i in range(training_N):
        # phi[:,i] = training_data[:,i] - avg
# a = np.matmul(phi.T, best_eigvecs) #Omega
# P = np.matmul(a,best_eigvecs_lowdim.T) 
# P = P.T
# eigfaces = np.zeros(P.shape) 

# for i in range(P.shape[1]):
        # eigfaces[:,i]=avg+P[:,i]

# """eigface_5 = avg + P.T[:,5]
# eigface_5 = eigface_5.astype(int)
# print_img(eigface_5)
# print_img(training_data[:,5])"""
# #---------------------------------------------------------------------------------------------------------------- Testing
# omega_n = a.T
# phi_test = np.zeros(testing_data.shape)

# for i in range(testing_N):
    # phi_test[:,i] = testing_data[:,i] - avg

# omega = np.matmul(phi_test.T,best_eigvecs_lowdim)
# omega = omega.T

# #----------------------------------------------------------------------------------------------- Nearest Neighbour method
# error_vec = np.zeros((1,testing_N))
# identity_vec = np.zeros((1,testing_N))
# predicted_labels = np.zeros((1,testing_N))

# for i in range(testing_N):
        # error_vec[0,i], identity_vec[0,i] = NN(omega[:,i],omega_n)

# identity_vec = identity_vec.astype(int)

# for i in range(testing_N):
        # p = identity_vec[0,i]
        # predicted_labels[0,i] = training_labels[0,p]

# #print(confusion_matrix(testing_labels.T, predicted_labels.T))

# #----------------------------------------------------------------------------------------------------- Alternative method
# erreur = alternative(training_data,testing_data)

# #Testing out the alternative method
# """p = 0
# labela = np.argmin(erreur[:,20]) #image 10 testing data label
# print_img(testing_data[:,20])


# for i in range(training_N):
        # if training_labels[0,i] == labela+1:
                # p = i
                # break
# print_img(training_data[:,p])"""

# #---------------------------------------------------------------------------------------------------------- PCA-LDA method
# pca_eigvals, wpca = low_dim_PCA(training_data,313) 

# omega_pca = np.matmul(phi.T, wpca) #Datapoints in PCA feature space
# omega_pca = omega_pca.T

# Sb, Sw = SbSw_calculation(omega_pca,training_labels)

# lda_eigvals, wlda = LDA(Sw, Sb, 25)  #Applying LDA

# #DOES THE PCA DATA NEED TO BE NORMALISED??
# train_proj = np.matmul(wlda.T,omega_pca) #Classification space projection for training data

# """for i in range(P_fish.shape[1]):
        # fisherfaces[:,i]=avg+P_fish[:,i]"""

# """a = np.matmul(phi.T, best_eigvecs) #Omega
# P = np.matmul(a,best_eigvecs_lowdim.T) 
# P = P.T
# eigfaces = np.zeros(P.shape) 

# for i in range(P.shape[1]):
        # eigfaces[:,i]=avg+P[:,i]"""


#-------------------------------------- PCA-LDA -------------------------------------------- 

#------------------------------- accuracy vs mpca, mlda ------------------------------------ 

testing_N = testing_data.shape[1]
training_N = training_data.shape[1]

avg = np.average(training_data, 1)
avg = avg[:,np.newaxis]
phiTrain = training_data - avg
phiTest = testing_data - avg
eigval, wPca = low_dim_PCA(training_data, 100)
omegaTrain = np.matmul(wPca.T,phiTrain)

for j in range(1,52):
	Sb, Sw = SbSw_calculation(omegaTrain,training_labels)
	lda_eigvals, wLda = LDA(Sw, Sb, j)

plt.figure()	
plt.plot(lda_eigvals)
plt.ylim(bottom=0)
plt.title('LDA Eigenvalues for MPca = 100')
plt.xlabel('Eigenvector')
plt.ylabel('Eigenvalue')
plt.show()
	
simulateData = False
if simulateData == True:
	Mpca = [1,2,3,4,5,10,15,20,50]
	#Mpca = np.concatenate((np.arange(1,10), np.arange(10,250,10)))
	Mlda = [1,2,3,4,5,10,20,50]
	#Mlda = np.concatenate((np.arange(1,4), np.arange(4,50,2)))

	avg = np.average(training_data, 1)
	avg = avg[:,np.newaxis]
	phiTrain = training_data - avg
	phiTest = testing_data - avg

	acc = np.zeros((len(Mpca),len(Mlda)))
	neigh = KNeighborsClassifier(n_neighbors=1) #kNN with 5 nearest neighbours considered

	for i in range(len(Mpca)):
		eigval, wPca = low_dim_PCA(training_data, Mpca[i])
		omegaTrain = np.matmul(wPca.T,phiTrain)
		omegaTest = np.matmul(wPca.T,phiTest)
		
		for j in range(len(Mlda)):
			Sb, Sw = SbSw_calculation(omegaTrain,training_labels)
			lda_eigvals, wLda = LDA(Sw, Sb, Mlda[j])  #Applying LDA
			print('Mpca = ', Mpca[i])
			print('Mpca = ', Mlda[j])
			print('Rank of Sw: ', matrix_rank(Sw))
			print('Rank of Sb: ', matrix_rank(Sb))
			train_proj = np.matmul(wLda.T,omegaTrain)
			test_proj = np.matmul(wLda.T,omegaTest) #Project that data to feature space

			neigh.fit(train_proj.T, np.ravel(training_labels)) #fit to data
			#predicted_labels_lda = neigh.predict(test_proj.T) #predict classes; returns shape [n_samples]
			acc[i,j] = neigh.score(test_proj.T,np.ravel(testing_labels)) #compute error

	#something weird happens with the rank until 51 (c-1)
	#rank(Sb) = rank(Sw) = 1,2,3,4...
	matdata = {
		"Mpca": Mpca,
		"Mlda": Mlda,
		"Accuracy": acc,
	}
	#sio.savemat('LdaAccuracy.mat', matdata)	

plotData = False
if plotData == True:
	mat_content = sio.loadmat('LdaAccuracy.mat')
	Mpca = mat_content['Mpca']
	Mlda = mat_content['Mlda']
	acc = mat_content['Accuracy']

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')	
	X, Y = np.meshgrid(Mpca, Mlda)	
	surf = ax.plot_surface(X, Y, acc.T, cmap=cm.coolwarm, linewidth=0, antialiased=False)
	fig.colorbar(surf, shrink=0.5, aspect=5)
	plt.title('Recognition Accuracy as a function of MPca, MLda')
	plt.xlabel('MPca')
	plt.ylabel('MLda')
	plt.show()

#--------------------------------- confusion matrices ------------------------------------ 

#from now on Mpca set as 100 and Mlda set as 25
#neigh = KNeighborsClassifier(n_neighbors=1) #kNN with 5 nearest neighbours considered
eigval, wPca = low_dim_PCA(training_data, 100)
avg = np.average(training_data, 1)
avg = avg[:,np.newaxis]
phiTrain = training_data - avg
phiTest = testing_data - avg
omegaTrain = np.matmul(wPca.T,phiTrain)
omegaTest = np.matmul(wPca.T,phiTest)

Sb, Sw = SbSw_calculation(omegaTrain,training_labels)
lda_eigvals, wLda = LDA(Sw, Sb, 25)
train_proj = np.matmul(wLda.T,omegaTrain)
test_proj = np.matmul(wLda.T,omegaTest) #Project that data to feature space

#neigh.fit(train_proj.T, np.ravel(training_labels)) #fit to data
#predicted_labels_lda = neigh.predict(test_proj.T) #predict classes; returns shape [n_samples]

error_lda = np.zeros((1,testing_N))
identity_lda = np.zeros((1,testing_N))
predicted_labels_lda = np.zeros((1,testing_N))

for i in range(testing_N):
        error_lda[0,i], identity_lda[0,i] = NN(test_proj[:,i],train_proj)

identity_lda = identity_lda.astype(int)

for i in range(testing_N):
        p = identity_lda[0,i]
        predicted_labels_lda[0,i] = training_labels[0,p]

cmData = confusion_matrix(testing_labels.T, predicted_labels_lda.T)
plt.figure()
plot_confusion_matrix(cmData,normalize=False,
                      title='Confusion matrix for PCA-LDA face recognition')
plt.show()

#---------------------------------- failure example ------------------------------------- 
"""
#misclassified example is for class 1

fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')

#Class 1
idx = np.where(training_labels == 1)[1]
c1 = train_proj[:,idx]
X1 = c1[0,:] 
Y1 = c1[1,:]
Z1 = c1[2,:] 
ax.scatter(X1,Y1,Z1,c='b',marker='o',label='Class 1') 

#Class 7
idx = np.where(training_labels == 7)[1]
c7 = train_proj[:,idx]
X7 = c7[0,:] 
Y7 = c7[1,:]
Z7 = c7[2,:] 
ax.scatter(X7,Y7,Z7,c='r',marker='o',label='Class 7') 

#Test example - predicted class 1, actual class 7
idx = np.where(testing_labels == 7)[1]
T1 = test_proj[0,20] 
T2 = test_proj[1,20] 
T3 = test_proj[2,20]  
ax.scatter(T1,T2,T3,c='k',marker='*',label='Failure: Predicted 1, Actual 7') 

T1 = test_proj[0,2] 
T2 = test_proj[1,2] 
T3 = test_proj[2,2]
ax.scatter(T1,T2,T3,c='k',marker='x',label='Success: Predicted 1, Actual 1') 

ax.set_xlabel('u1')
ax.set_ylabel('u2')
ax.set_zlabel('u3')
ax.legend()
plt.show()

plt.title('Class 7 Image misclassified as Class 1 with LDA')
print_img(testing_data[:,20])
plt.title('Nearest match (Class 1)')
print_img(training_data[:, identity_lda[0,20]])
plt.title('Class 1 Image correctly classified with LDA')
print_img(testing_data[:,2])
plt.title('Nearest match (Class 1)')
print_img(training_data[:, identity_lda[0,2]])
"""

#---------------------------------- RANDOM SAMPLING ON TRAINING DATA ------------------------------------ 
simulateData = False
if simulateData == True:
	T = [1,3,4,5,6,7,8,9,10,20,50]
	c1 = [10, 20, 30, 40, 50]
	acc = np.zeros((len(c1),len(T)))
	alpha = np.zeros((len(c1),len(T)))

	for j in range(len(c1)):
		for i in range(len(T)):
			ypredEnsemble, ytrue, errorCom, errorAvg = dataEnsemblePCA(training_data, training_labels, testing_data, testing_labels, Mpca=100, Mlda=25, c1=c1[j], T=T[i])
			acc[j,i] = 1-errorCom
			alpha[j,i] = errorCom/errorAvg


	matdata = {
		"Recognition Accuracy": acc,
		"Alpha": alpha,
		"T": T,
		"c1": c1,
	}
	sio.savemat('BootstrapData.mat', matdata)	

plotData = False
if plotData == True:
	mat_content = sio.loadmat('BootstrapData.mat')
	acc = mat_content['Recognition Accuracy']
	alpha = mat_content['Alpha']
	T = mat_content['T']
	c1 = mat_content['c1']
	T = T.T

	plt.figure(1)
	for i in range(c1.shape[1]):
		plt.plot(T, acc[i,:])
	plt.title('Recognition Accuracy against Number of Bootstrap Replicates')
	plt.ylabel('Recognition Accuracy')
	plt.xlabel('Replicates per Ensemble')
	plt.legend(['c1 = 10', 'c1 = 20', 'c1 = 30', 'c1 = 40', 'c1 = 50'])
	plt.tight_layout()

	plt.figure(2)
	for i in range(c1.shape[1]):
		plt.plot(T, alpha[i,:])
	plt.title('Alpha against Number of Bootstrap Replicates')
	plt.ylabel('Alpha')
	plt.xlabel('Replicates per Ensemble')
	plt.legend(['c1 = 10', 'c1 = 20', 'c1 = 30', 'c1 = 40', 'c1 = 50'])
	plt.tight_layout()	
	plt.show()

				  
#---------------------------------- RANDOM SAMPLING IN FEATURE SPACE ------------------------------------ 

simulateData = False
if simulateData == True:
	T = [1, 3, 6, 10, 20, 50]
	M0 = [10, 50, 100]
	M1 = [10, 20, 50]
	acc = np.zeros((len(M0),len(M1),len(T)))
	alpha = np.zeros((len(M0),len(M1),len(T)))
	for k in range(len(M0)):
		for j in range(len(M1)):
			for i in range(len(T)):
				ypredEnsemble, ytrue, errorCom, errorAvg = featureEnsemble(training_data, training_labels, testing_data, testing_labels, Mlda=25, M0=M0[k], M1=M1[j],T=T[i])
				acc[k,j,i] = 1-errorCom
				alpha[k,j,i] = errorCom/errorAvg

	matdata = {
		"Recognition Accuracy": acc,
		"Alpha": alpha,
		"T": T,
		"M0": M0,
		"M1": M1,
	}
	sio.savemat('FeatureData.mat', matdata)	

plotData = False
if plotData == True:
	mat_content = sio.loadmat('FeatureData.mat')
	acc = mat_content['Recognition Accuracy']
	alpha = mat_content['Alpha']
	T = mat_content['T']
	M0 = mat_content['M0']
	M1 = mat_content['M1']
	T = T.T


	for j in range(M0.shape[1]):
		plt.figure(j)
		for i in range(M1.shape[1]):
			plt.plot(T, acc[j,i,:])
			
		plt.title('Recognition Accuracy against M1, M0 = %i' %M0[0,j])
		plt.ylabel('Recognition Accuracy')
		plt.xlabel('Replicates per Ensemble')
		#plt.legend(['M0 = %i' %M0[0,0], 'M0 = %i' %M0[0,1], 'M0 = %i' %M0[0,2], 'M0 = %i' %M0[0,3], 'M0 = %i' %M0[0,4]])
		plt.legend(['M1 = %i' %M1[0,0], 'M1 = %i' %M1[0,1], 'M1 = %i' %M1[0,2]])
		plt.tight_layout()
		
	for j in range(M0.shape[1]):
		plt.figure()
		for i in range(M1.shape[1]):
			plt.plot(T, alpha[j,i,:])
			
		plt.title('Alpha against M1, M0 = %i' %M0[0,j])
		plt.ylabel('Alpha')
		plt.xlabel('Replicates per Ensemble')
		#plt.legend(['M0 = %i' %M0[0,0], 'M0 = %i' %M0[0,1], 'M0 = %i' %M0[0,2], 'M0 = %i' %M0[0,3], 'M0 = %i' %M0[0,4]])
		plt.legend(['M1 = %i' %M1[0,0], 'M1 = %i' %M1[0,1], 'M1 = %i' %M1[0,2]])
		plt.tight_layout()
		
	plt.show()
	
#------------------------------------------- FUSION RULES ----------------------------------------- 

simulateData = True
if simulateData == True:
	alpha = np.zeros((1,3))

	ypredEnsemble, ytrue, errorCom, errorAvg = featureEnsemble(training_data, training_labels, testing_data, testing_labels, Mlda=25, M0=50, M1=50, T=20, fusiontype='majority')
	alpha[0,0] = errorCom/errorAvg
	ypredEnsemble, ytrue, errorCom, errorAvg = featureEnsemble(training_data, training_labels, testing_data, testing_labels, Mlda=25, M0=50, M1=50, T=20, fusiontype='sum')
	alpha[0,1] = errorCom/errorAvg
	ypredEnsemble, ytrue, errorCom, errorAvg = featureEnsemble(training_data, training_labels, testing_data, testing_labels, Mlda=25, M0=50, M1=50, T=20, fusiontype='product')
	alpha[0,2] = errorCom/errorAvg

	print(alpha)

#---------------------------------------- CONFUSION MATRICES -------------------------------------- 
ypredEnsemble, ytrue, errorCom, errorAvg = featureEnsemble(training_data, training_labels, testing_data, testing_labels, Mlda=25, M0=50, M1=50, T=20, fusiontype='majority')
cmFeatures = confusion_matrix(ytrue.T, ypredEnsemble.T)
plt.figure()
plot_confusion_matrix(cmFeatures,normalize=False,
                      title='Confusion matrix for Random Feature Ensemble')

ypredEnsemble, ytrue, errorCom, errorAvg = dataEnsemblePCA(training_data, training_labels, testing_data, testing_labels, Mpca=100, Mlda=25, c1=30, T=20, fusiontype='majority')
cmFeatures = confusion_matrix(ytrue.T, ypredEnsemble.T)
plt.figure()
plot_confusion_matrix(cmFeatures,normalize=False,
                      title='Confusion matrix for Bagging Ensemble')
plt.show()
