import numpy as np
import scipy.io as sio
import time
from sklearn.cluster import KMeans

class DownSampler:
    """ A Python implementation of a k-medioids clusterer """
    def __init__( self ):
        """
        Constructor

        Parameters
        ----------
        X : numpy matrix of vectors to downsample features x samples
            
        k : downsample to k elements

        Returns
        -------
        obj
            Downsampled matrix features x samples
            indecies of selected vectors 
        """
        pass

    def mean_uniform(self, X, k = 5):
        """
        Performs mean uniform downsampling

        Parameters
        ----------
        X : numpy.ndarray (n_samples, n_features) 
            The data to downsample
        k : int
            The number of samples after downsampling

        Returns
        -------
        numpy.ndarray (k, n_features)
            Downsampled data
        """
        X = X.T

        shape_of_X = X.shape
        
        sample_hopp = (shape_of_X[1]-1)/(k-1)

        indecies = np.zeros(k+1, dtype=int)
        for i in range(k+1):
            indecies[i] = int(round(sample_hopp*(i)))

        X_out = np.empty([shape_of_X[0],k])

        for i in range(k):
            X_out[:,i] = np.mean(X[:,indecies[i]:indecies[i+1]],1)

        return X_out.T

    def uniform( self, X, k = 5 ):
        """
        Performs mean uniform downsampling

        Parameters
        ----------
        X : numpy.ndarray (n_samples, n_features) 
            The data to downsample
        k : int
            The number of samples after downsampling

        Returns
        -------
        numpy.ndarray (k, n_features)
            Downsampled data
        numpy.ndarray (k,)
            Indices of downsampled data
        """
        
        X = X.T

        shape_of_X = X.shape
        
        sample_hopp = (shape_of_X[1]-1)/(k-1)

        indecies = np.zeros(k+1, dtype=int)
        for i in range(k+1):
            indecies[i] = int(round(sample_hopp*(i)))

        X_out = np.empty([shape_of_X[0],k])

        for i in range(k):
            X_out[:,i] = X[:,indecies[i]]
        
        return X_out.T, indecies

    def uniform_cache( self, X, k = 5 ):
        """
        Performs mean uniform downsampling

        Parameters
        ----------
        X : numpy.ndarray (n_samples, n_features) 
            The data to downsample
        k : int
            The number of samples after downsampling

        Returns
        -------
        numpy.ndarray (k, n_features)
            Downsampled data
        numpy.ndarray (k,)
            Indices of downsampled data
        """
        shape_of_X = len(X)
        
        sample_hopp = (shape_of_X)/(k)

        indecies = np.zeros(k, dtype=int)
        for i in range(k):
            indecies[i] = int(round(sample_hopp*(i)))

        for i in range(k):
            X[i] = X[indecies[i]]
        
        return X

    def k_medoids( self, X, k = 5 ):
        """
        Performs k_medoids PAM downsampling

        Parameters
        ----------
        X : numpy.ndarray (n_features, n_samples) 
            The data to downsample
        k : int
            The number of samples after downsampling

        Returns
        -------
        numpy.ndarray (k,)
            Indices of downsampled data
        """
        shape_of_X = X.shape

        dot_product = []
        for i in range(shape_of_X[1]):
            v = 0
            for j in range(shape_of_X[0]):
                v = v + (X[j,i] * X[j,i].T)

            dot_product.append(v)
        
        dot_product = np.asarray(dot_product).reshape(1,shape_of_X[1])
        test = ( dot_product - 2 * ( X.T @ X ) )
        D = dot_product.T + ( dot_product - 2 * ( X.T @ X ) )
        np.fill_diagonal(D, 0)

        cost = np.zeros( k )
        medoid_assign = np.zeros( shape_of_X[1] )

        medoid_indecies = []

        for i in range( k ):
            medoid_indecies.append( np.round((shape_of_X[1]/k)*i))
        
        medoid_indecies = np.unique(medoid_indecies)

        medoid_indecies = np.asarray(medoid_indecies).reshape(1,k)
        medoid_indecies = medoid_indecies.astype( 'int32' )


        for i in range( k ):
            t_mt = time.perf_counter()
            if i == 0:

                for j in range( shape_of_X[1] ):
                    min_val = min(D[j,medoid_indecies][0])
                    min_idx = np.where(D[j,:] == min_val)
                    medoid_assign[j] = min_idx[0][0]

                for j in range( shape_of_X[1] ):
                    
                    if np.where(medoid_indecies == j)[-1].size == 0:
                        cost_idx = np.where(medoid_indecies == medoid_assign[j])[-1]
                        cost[ cost_idx ] = cost[ cost_idx ] + D[ j, int(medoid_assign[j]) ]
            

            for j in range( shape_of_X[1] ):
                tk = time.perf_counter()
                medoid_indecies_temp = medoid_indecies.copy()

                if np.where(medoid_indecies == j)[-1].size == 0:
                    cost_temp = np.zeros( k )

                    medoid_indecies_temp[0][i] = j
                    
                    for l in range( shape_of_X[1] ):
                        min_val = min(D[l,medoid_indecies_temp][0])
                        min_idx = np.where(D[l,:] == min_val)
                        medoid_assign[l] = min_idx[0][0]

                    for l in range( shape_of_X[1] ):
                    
                        if np.where(medoid_indecies_temp == l)[-1].size == 0:
                            cost_idx = np.where(medoid_indecies_temp == medoid_assign[l])[-1]                                                     
                            cost_temp[ cost_idx ] = cost[ cost_idx ] + D[ l, int(medoid_assign[l]) ]

                    if np.sum( cost_temp ) < np.sum( cost ):
                        cost = cost_temp.copy()
                        medoid_indecies = medoid_indecies_temp.copy()
                tf = time.perf_counter()
                #print(f"Inner: {tf-tk}")
            tj = time.perf_counter()
            #print(f"Outer: {tj-t_mt}")
        return medoid_indecies

    def k_means(self, X, k = 5 ):

        kmeans = KMeans(n_clusters=k, random_state=0).fit(X.T)

        return kmeans.cluster_centers_

if __name__ == '__main__':
    
    downsample = DownSampler()

    X = sio.loadmat('_transactional_data\init.mat')['init']['Xtrain'][0][0].T
    L = sio.loadmat('_transactional_data\init.mat')['init']['ytrain'][0][0]
    class_idx = np.where(L == 2)
    class_to_downsample = X[ :, class_idx[-1] ]

    downsample.k_medoids( class_to_downsample )    

    test_mat = np.zeros( shape = ( 2000, 10 ) )
    for i in range( 2000 ):
        test_mat[i,:] = i
        
    downsample = DownSampler()
    X_uniform, I = downsample.uniform( test_mat, 5 )
    X_mean_uniform = downsample.mean_uniform( test_mat, 5 )
    
    print( 'X Uniform:\n', X_uniform )
    print( 'I:', I )
    print( 'X Mean Uniform:\n', X_mean_uniform )