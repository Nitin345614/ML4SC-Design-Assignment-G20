def rbf (x1, x2, width):
    return np.exp(- (np.sum((x1 - x2)**2))/width)

def rbf_clip (x1, x2, width):
    euc = np.mean((x1 - x2)**2)
    if euc>width:
        return 0
    else:
        return np.exp(- euc/width)

def uniform (x1, x2, width):
    return (np.mean((x1 - x2)**2)>=width)*0.5

def slice(x1,x2,kernel,width,axis=1):
    n1 = x1.shape[0]
    n2 = x2.shape[0]
    dim = x1.shape[1]

    if dim != x2.shape[1]:
        print("incompatible input dimensions")
        return
    if (n1!=n2)&(axis==0):
        print("incompatible input lengths")
        return

    if axis==0:
        out = np.zeros([n1])
        for i in range(n1):
            out[i] = kernel(x1[i,:],x2[i,:],width)
        return out
    else:
        out = np.zeros([n1, n2])
        for i in range(n1):
            for j in range(n2):
                out[i,j] = kernel(x1[i,:],x2[j,:],width)
        return out

def Kcomp(x1,x2,width,kernel,project=False,included=0):
    if project==True:
        xm = x1[included]
        Kmm = slice(xm,xm,kernel,width)
        Km1 = slice(xm,x1,kernel,width)
        #K1m = Km1.T
        Km2 = slice(xm,x2,kernel,width)
        #K2m = Km2.T
        K22 = slice(x2,x2,kernel,width)
        return Kmm,Km1,Km2,K22
    else:
        K11 = slice(x1,x1,kernel,width)
        K12 = slice(x1,x2,kernel,width)
        #K21 = K12.T
        K22 = slice(x2,x2,kernel,width)
        return K11,K12,K22

def post(x1,x2,y,width,kernel,noise,environment,project=False,included=None):
    N = len(x1)
    if project==True:
        Kmm,Km1,K1m,Km2,K2m,K22 = Kcomp(x1,x2,width,kernel,project,included)
        f = Km2.T@np.linalg.inv(noise**2 *Kmm + Km1@Km1.T)@Km1@y
        
        Kmm_inv = np.linalg.inv(Kmm)
        err_inv = np.linalg.inv(Kmm + Km1@K1m/(noise**2))
        cov = K22 - Km2.T@(Kmm_inv - err_inv)@Km2
        return f,cov

    else:
        #K11,K12,K22 = Kcomp(x1,x2,width,kernel,project)
        #K11_inv = np.linalg.inv(K11 + np.eye(N)*noise**2)
        #A = K12.T@K11_inv
        #f = A@y
        #cov = K22 - A@K12

        reg = GaussianProcessRegressor(kernel,alpha=environment.alpha,n_restarts_optimizer=environment.n_restarts_optimizer) 
        reg.fit(x1,y)
        f, cov = reg.predict(x2,return_cov=True)

        #f = K12.T@K11_inv@y
        #cov = K22 - K12.T@K11_inv@K12
        return f,cov,reg


def MSLL(y1,y2,noise,var,project=False,included=0):
    if y1.ndim > 1: y1 = y1[:,0]
    if y2.ndim > 1: y2 = y2[:,0]
    if var.ndim > 1: var = var[:,0]

    star = noise**2 + var
    log_loss = ( 0.5 * np.log(2*np.pi*star)) + ( (y1 - y2)**2 / (2*star))

    if project==False:
        simple_mean, simple_var =  np.mean(y1), np.std(y1)**2
    else:
        simple_mean, simple_var =  np.mean(y1[included]), np.std(y1[included])**2

    simple_star = noise**2 + simple_var
    simple_loss = ( 0.5 * np.log(2*np.pi*simple_var)) + ( (y1 - simple_mean)**2 / (2*simple_var))
    return np.mean(log_loss - simple_loss)

def sd_algo(x1,x2,y,M_sd,width,kernel,noise,environment,testiness=0.5):
    N = y.shape[0]
    included = np.array(np.zeros(N),dtype="bool")
    i = np.random.randint(low=0,high=N-1)
    included[i] = True

    testpoints = np.random.uniform(low=0,high=1,size=[N]) > (1-testiness)

    for m in tqdm(range(M_sd)):
        f,cov,reg = post(x1[included,:],x2[testpoints,:],y1[included,:],width,kernel,noise,environment)
        var = np.diag(cov)
        i_new = np.argmax(var)
        included[i_new] = True
    
    if environment.quiet==False: return f,var,included,cov,testpoints,reg
    if environment.quiet==True: return f,included,cov,reg

def ARX_load_input(u,y,batches=1,order=1):
    n = u.shape[0]
    N = int(n/batches)
    ux = np.concat(([0],u),axis=0)
    yx = np.concat(([0],y),axis=0)

    ux_temp = np.zeros([n,order])
    yx_temp = np.zeros([n,order])

    for i in range(batches):
        ux_temp[i*N:((i+1)*N),:] = scipy.linalg.toeplitz(u[i*N:((i+1)*N)])[:,0:order]
        yx_temp[i*N:((i+1)*N),:] = scipy.linalg.toeplitz(y[i*N:((i+1)*N)])[:,0:order]

    ux = np.tril(ux_temp,k=-1)[0:n,0:order]
    yx = np.tril(yx_temp,k=-1)[0:n,0:order]
    
    return np.concat((ux,yx),axis=1)

def demonstrate(full_range,y1,f,var,testpoints,name):
    plt.figure(figsize=(16,4.8))
    plt.title(name)
    plt.plot(full_range,y1,label="Data",marker=".")
    plt.plot(full_range[testpoints],f,label="Prediction")
    plt.fill_between(full_range[testpoints],f+var,f-var,alpha=0.3,label='est std out')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()
    return

def quick_plot(full_range,f,var,name):
    plt.figure(figsize=(16,4.8))
    plt.title(name)
    plt.plot(full_range,f,label="Prediction")
    plt.fill_between(full_range,f+var,f-var,alpha=0.3,label='est std out')
    plt.xlabel('x')
    plt.ylabel('f')
    plt.legend()
    plt.show()
    return