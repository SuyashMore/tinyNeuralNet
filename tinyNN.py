import numpy as np


# Activation Functions

def activation_sigmoid(X,der=False):
    if not der:
        return np.divide(1, 1 + np.exp(-X) )
    else:
        return np.multiply(X,(1-X))

def activation_linear(X,der=False):
    if not der:
        return X
    else:
        return np.ones(X.shape)


def activation_relu(X,der=False):
    if not der:
        return np.maximum(0,X)
    else:
        return (X>=0).astype(float)

def activation_softmax(X,der=False):
    if not der:
        expp = np.exp(X-np.max(X,axis=0))
        denum = np.sum(expp,axis=0)
        res = np.divide(expp,denum)
        return res
    else:
        print("Something is Wrong")
        pass


# Loss Functions

def crossEntropyCost(Y,Y_orig,der=False):
    if not der:
        m=Y.shape[1]
        costMat = Y_orig*np.log(Y)+(1-Y_orig)*np.log(1-Y)
        print(f"Cost Mat:\n{costMat}")
        cost = -np.sum(costMat)/m
        cost = np.squeeze(cost)
        assert(cost.shape == ())
        return cost
    else:
        dLda = -(np.divide(Y_orig, Y) - np.divide(1 - Y_orig, 1 - Y))  
        return dLda


def cross_entropy(Y, Y_orig, epsilon=1e-12):
    """
    Computes cross entropy between targets (encoded as one-hot vectors)
    and predictions. 
    Input: predictions (N, k) ndarray
           targets (N, k) ndarray        
    Returns: scalar
    """
    predictions = np.clip(Y, epsilon, 1. - epsilon)
    N = Y.shape[0]
    ce = -np.sum(Y_orig*np.log(Y+1e-9))/N
    return ce


# Network Class

class NeuralNetwork():
    def __init__(self):
        self.layerSizes = []
        self.activations = []

    def debugPrint(self,x):
        if self.debug:
            print(x)

    def init_adam(self,beta1 = 0.9,beta2=0.999):
        #Adam Optimizer Terms
        self.v = [None]*self.totalLayers #Momentum
        self.s = [None]*self.totalLayers #RMSProp
        self.v_corrected = [None]*self.totalLayers #Momentum
        self.s_corrected = [None]*self.totalLayers #RMSProp

        self.beta1 = beta1
        self.beta2 = beta2

        for i in range (self.totalLayers):

            # Dimensions of Weight Matrix : (layer i+1 , layer i),Bias:(layer i,1)
            self.v[i]=np.zeros(self.weights[i].shape)
            self.s[i]=np.zeros(self.weights[i].shape)
            self.v_corrected[i]=np.zeros(self.weights[i].shape)
            self.s_corrected[i]=np.zeros(self.weights[i].shape)
            


    def compile(self,lr=0.9,debug=False):

        self.debug=debug
        self.lr = lr

        # Initialize Empty Arrays to Store Weights and Biases
        self.totalLayers = len(self.layerSizes)-1
        self.weights = [None]*self.totalLayers
        self.bias = [None]*self.totalLayers

        

        # Initialize Weight and Bias Matrix 
        for i in range (len(self.layerSizes)-1):

            # Dimensions of Weight Matrix : (layer i+1 , layer i),Bias:(layer i,1)

            #He Random Initialization
            self.weights[i]=np.random.random((self.layerSizes[i+1],self.layerSizes[i]))*((2/(self.layerSizes[i]))**0.5)
            self.bias[i]=np.random.random((self.layerSizes[i+1],1))*0.1
            
            assert(self.weights[i].shape==(self.layerSizes[i+1],self.layerSizes[i]))
            assert(self.bias[i].shape==(self.layerSizes[i+1],1))

        
        print("Model Compiled Successfully")

    def addLayer(self,layerSize,activation=None):
        self.layerSizes.append(layerSize)

        if len(self.layerSizes)==1: # No Activation on Input Layer
            assert(activation==None) 
        else:
            assert(activation!=None) # Activation Required
            self.activations.append(activation)

    
    def forward(self,X):
        Z = None
        A = X
        self.activationCache = [] # Used to Compute Gradients

        self.debugPrint("=================Model Meta=====================")
        self.debugPrint(f"Layers(Including Input) :{self.layerSizes}")
        self.debugPrint(f"Activations:{self.activations}")
        self.activationCache.append(X)
        # Forward Propogate through Each Layer
        for i in range(self.totalLayers):
            Z = np.dot(self.weights[i],A)+self.bias[i] # Z= W.X + b
            
            A = self.activations[i](Z)                  # A = activation(Z)
            self.activationCache.append(A)

            self.debugPrint("======================================================")
            self.debugPrint(f"Layer #{i+1} Output:\n{A}")
            self.debugPrint(f"Layer Weights:\{self.weights[i]}")
            self.debugPrint(f"Layer Bias:\{self.bias[i]}")
        self.debugPrint("================== End Of Model =================================")
        self.last_Output = A
        return A


    def fit(self,X,Y,epochs=1):
        # Feed Forward
        self.init_adam()

        result = self.forward(X)
        
        #Compute Cost
        cost = cross_entropy(result,Y)
        print(f"Initial Cost:{cost}")

        for i in range(epochs):
            # Feed Forward
            # result = self.forward(X)

            #Compute Gradients
            dWs,dbs = self.compute_grads(Y,result)

            #Apply Gradients
            self.apply_grads_adam(dWs,dbs,i+1)

            # Compute Cost
            result = self.forward(X)

            cost = cross_entropy(result,Y)

            # print(f"Result:\n{result} \n Y:\n{Y}  \n cost:{cost}")

            print(f"Epoch ({i}/{epochs-1})=> Cost:{cost}")



    #Use Standard Gradient Descent
    def apply_grads(self,dWs,dbs,t):

        for i in range(self.totalLayers):
            self.weights[i] -= self.lr * dWs[i]
            self.bias[i] -= self.lr * dbs[i]

    # Used Adam Optimizer
    def apply_grads_adam(self,dWs,dbs,t):
        
        for i in range(self.totalLayers):
            self.v[i]= self.beta1*self.v[i]+(1-self.beta1)*dWs[i]
            self.v_corrected[i]= np.divide(self.v[i],(1-self.beta1**t))
            # print(f"V-corrected:\n{self.v_corrected[i]}")
            # self.s[i]= self.beta2*self.s[i]+(1-self.beta2)*np.power(dWs[i],2)
            # self.s_corrected[i]= np.divide(self.s[i],(1-self.beta2**t))

            numerator=self.v_corrected[i]
            # denum = np.power(self.s[i],0.5)+1e-8

            # Currently use only Momentum
            dW_adam = np.divide(numerator,1)
            # dW_adam = self.v[i]
            # print(f"Dw_adam:\n{dW_adam}")

            # print(f"Dw_adam:\n{dW_adam}")
            self.weights[i] -= self.lr * dW_adam
            self.bias[i] -= self.lr * dbs[i]



    def compute_grads(self,Y,result): # Returns Cost and Grads(dws,dbs)

        def linear_backward(dZ,cached_activation,weights,bias): #Calculate grads for Layer i
            #cached_activation(activation in layer i),weights(weight Matrix of ith layer),bias(bias matrix of ith layer)
            m=cached_activation.shape[1] #For Vetorized Implementation
            # print(f"dZ = {dZ}")
            # print(f"Cached Act:{cached_activation}")
            dW = np.dot(dZ,cached_activation.T)/m
            db = np.sum(dZ,axis=1,keepdims=True)
            dA = np.dot(weights.T,dZ)
            # print(f"dW = {dW}")

            assert (dW.shape == weights.shape)
            assert (db.shape == bias.shape)
            assert (dA.shape == cached_activation.shape)
            

            return (dW,db,dA)

        
        # dLda = cross_entropy(self.last_Output,Y,der=True)


        dZs = [None]*self.totalLayers
        dWs = [None]*self.totalLayers
        dbs = [None]*self.totalLayers
        

        # dZs[-1]=dLda
        dldz=None
        dadz=None
        layer_cache=None
        # print(f"dZs:{dZs}")
        for l in reversed(range(self.totalLayers)):
            # print(f"Layer #{l} ")
            if l==(self.totalLayers-1):
                dldz = result-Y
                layer_cache = self.activationCache[l]
            else:
                dadz = self.activations[l](self.activationCache[l+1],der=True)
                layer_cache = self.activationCache[l]
                dldz = np.multiply(dZs[l],dadz)
            dw,db,dA = linear_backward(dldz,layer_cache,self.weights[l],self.bias[l])
            dZs[l-1]=dA
            dWs[l]=dw
            dbs[l]=db

        return dWs,dbs

