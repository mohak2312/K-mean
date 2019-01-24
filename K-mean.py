# To run the code, give the proper path for the training and testing file
# it will calculate results the for both experiments and print the results
# save the visulization images in the folder

# library used for the homewor
import numpy as np
from collections import Counter
import random as ra
import math
from sklearn.metrics import confusion_matrix                    # for confusion matrix
import matplotlib.pyplot as plt                                 # to ploat the images


file_train='optdigits.train'                                                      # file name for traing and testing                                        
file_test='optdigits.test'


#get the dataset for files
def get_dataset(file):
    data=[]
    with open(file,'r') as file:
        d1=file.read()                                      #read the file
        d=d1.splitlines()
        for values in d:
            x=values.split(",")
            x=[float(i) for i in x]
            data.append(x)
    data_frame= pd.DataFrame(data)
    y=data_frame[len(data[0])-1].values.tolist()            # create label for dataset
    X=data_frame.loc[:,0:len(data[0])-2].values.tolist()    # create feaure set
    return X,y

# function to calulate the euclidian distance
def cal_euclidean(train,test):
    distance=np.sqrt(np.sum(np.power(np.array(train)-np.array(test),2)))
    return distance

# function to train the training set
def KNN(X_train,Int_centeroid,y_train):
    
    lastcluster=[]
    while True:
        c=[]
        cluster={}
        label={}
        
        for a,train in enumerate(X_train):
            distance=[]
            
            for j,centeroid in enumerate(Int_centeroid):
                eu_distance=cal_euclidean(centeroid,train)                          # caculate the euclidian distance
                distance.append(eu_distance)
            
            c.append(np.argmin(np.array(distance)))
            cluster.setdefault(np.argmin(np.array(distance)),[]).append(train)      # create cluster dataset
            label.setdefault(np.argmin(np.array(distance)),[]).append(y_train[a])   # create label dataset for cluster
    
        if(np.array_equal(np.array(lastcluster),np.array(c)) == True):              # condition for otimization
            break
            
        else:
            lastcluster=c
            new_centeroid=cal_mean(cluster)
            Int_centeroid=np.array(new_centeroid)
            Int_centeroid=Int_centeroid.tolist()
            
    return Int_centeroid,cluster,label
# function to calculate the mean
def cal_mean(all_clusters):
    new_centeroid=[]
    for cluster in all_clusters:
        new_centeroid.append(np.mean(np.array(all_clusters[cluster]),axis=0))
    return new_centeroid


#functio to calulate the average mean swuare error

def cal_AMSE(Input,Mean):
    MSE=[]
    a=[]
    for i in range(len(Mean)):
        for j in range(len(Input[i])):
            mse=np.square(cal_euclidean(Input[i][j],Mean[i]))           #square the euclidian distance
            MSE.append(mse)
        a.append(np.mean(MSE))
    
    Avr_MSE=np.mean(np.array(a))
    return Avr_MSE

# function for mean square separation
def cal_MSS(mean,k):
    counter =0
    sum=0
    for i in range(len(mean)-1):
        for j in range(i+1,len(mean)):
            sum=sum+ np.sum(np.power(np.array(mean[i])-np.array(mean[j]),2))
            counter+=1
    MSS=sum/((k*(k-1))/2)
    return MSS
# function for mean entropu
def cal_mean_entropy(label,y_train):
    entropy_c=[]
    entropy=0
    mean_entropy=0
    for c in label:
        x=Counter(label[c])
        y=len(label[c])
        for i in x:
            entropy -= (x[i]/y)*math.log(x[i]/y,2)                      #entropy formula
        mean_entropy += entropy*(y/len(y_train))
    mean_entropy=mean_entropy/100
    return mean_entropy
        
#function for predicting the cluster class

def cal_class(label):
    cluster_class=[]
    for c in label:
        x=Counter(label[c])
        cluster_class.append(Counter(x).most_common(1)[0][0])
    return cluster_class  

# function or test
def KNN_test(X_test,Int_centeroid,cluster_class):
    
    predict_label=[]
    #print(cluster_class)
    for a,test in enumerate(X_test):
        distance=[]

        for j,centeroid in enumerate(Int_centeroid):
            eu_distance=cal_euclidean(centeroid,test)       # calculate the euclidian distance
            distance.append(eu_distance)
        #print(distance)
        a=np.argmin(np.array(distance))
        predict_label.append(cluster_class[a])              # predict the lable
    return predict_label

#function for accuracy calculation

def cal_accuracy(actual,predict):
    temp=0
    for i in range(len(actual)):                            # if predict=actual then increase the count
        if actual[i]==predict[i]:
            temp+=1
    accuracy=(temp/len(actual))*100
    return accuracy

# function for visulization
def visulize(new_centeroid,cluster_class,K):
    for j,i in enumerate(new_centeroid):  
        a=np.split(np.array(i),8)
        plt.imshow(a, cmap='gray_r')
        plt.title('cluster : {}'.format(j+1))
        plt.savefig("cluster_"+str(K)+"_"+str(j+1)+".jpeg")
        plt.show()



file_train='optdigits.train'                                                      # file name
X_train,y_train=get_dataset(file_train)                                           # get the training set
file_test='optdigits.test'                                                       # file name for test
X_test,y_test=get_dataset(file_test)                                             # get the test data
K_val=[10,30]                                                               # set the k value
for K in K_val:
    
    print("|--------------- Experiment for K="+str(K)+" ---------------|")
    cluster_N={}
    label_N={}
    new_centeroid_N=[]
    last_AMSE=0
    Avr_MSE_N=0
    for m in range(5):
        Int_centeroid=X_train
        ra.shuffle(Int_centeroid)                                           # suffle the dataset and select the K centroid
        Int_centeroid=Int_centeroid[:K]
        new_centeroid,cluster,label=KNN(X_train,Int_centeroid,y_train)      # run k-nn algorithm
        Avr_MSE=cal_AMSE(cluster,new_centeroid)                             # calculate avrage MSE
        print(Avr_MSE)
        if(m!=0):                                                           # condition for checkin min average MSE
            if(last_AMSE>=Avr_MSE):
                cluster_N=cluster
                label_N=label
                new_centeroid_N=new_centeroid
                Avr_MSE_N=Avr_MSE
                last_AMSE=Avr_MSE
                print("change")
        else:
            last_AMSE=Avr_MSE
            cluster_N=cluster
            label_N=label
            new_centeroid_N=new_centeroid
            Avr_MSE_N=Avr_MSE
    
    MSS=cal_MSS(new_centeroid_N,K)                                          # calculate the mean square separation
    mean_entropy=cal_mean_entropy(label_N,y_train)                          # calculate mean entropy
    print("Average mean square error : ",Avr_MSE_N)
    print("Mean square sepration : ",MSS)
    print("Mean entropy : ",mean_entropy)
    cluster_class=cal_class(label_N)                                        # predic the cluster class
    #print(cluster_class)
    predict_label=KNN_test(X_test,new_centeroid_N,cluster_class)            # k-nn algorithm on test set
    accuracy=cal_accuracy(y_test,predict_label)                             # calculate the accuracy
    print("Accuracy : ",accuracy)
    c=confusion_matrix(y_test, predict_label)                               #calculate the confusion matrix
    print("Confusion matrix :",c)
    visulize(new_centeroid_N,cluster_class,K)                               # visulization for cluster
    

