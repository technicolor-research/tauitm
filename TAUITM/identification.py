#########################################################################################################################################
## Copyright (c) 2016 - Technicolor R&D France
## 
## The source code form of this Open Source Project components is subject to the terms of the Clear BSD license.
##
## You can redistribute it and/or modify it under the terms of the Clear BSD License (http://directory.fsf.org/wiki/License:ClearBSD)
##
## See LICENSE file for more details.
##
## This software project does also include third party Open Source Software: See data/LICENSE file for more details.
#########################################################################################################################################

import numpy as np
from numpy import sqrt, random, array, divide, exp, pi, cos, sin, arctan2, ceil, floor, ones, zeros
from scipy.special import iv
import signal
import performance

def one_assignement(composite,movies,household_size=2):
    """ Assign everything to the same user."""
    assignments=[]
    for account in composite:
        classification=[]
        for show in account[1]:
            classification.append((show[0],0))
        assignments.append((account[0],classification))
    return assignments

def dirichlet_prediction(Y,n_pred,accounts,assign=True):
    Yuv_m=[y.sum(axis=1) for y in Y]
    predictions=[]
    for yuv_m,account in zip(Yuv_m[0:n_pred],accounts[0:n_pred]):
        clusters=[]
        for e in yuv_m.transpose():
            if assign:
                clusters.append(np.argmax(e))
            else:
                clusters.append(round(e[0]/(e[0]+e[1]),4))
        prediction = (account[0],[(film[0],cluster) for cluster, film in zip(clusters,account[1])])
        predictions.append(prediction)
    return predictions



def em_kabutoya(accounts,movies,mid_to_pos,composite=None,Y=None,Nb_it=100,verbose=False,alpha=0.001,beta=0.001,gamma=0.001,K=50,household_size=2,Time_Period=24.):
    if not composite:
        composite=accounts
        
    # Initialization
    V=household_size
    U=len(accounts) # Number of accounts
    I=len(movies) # Number of movies
    M=[len(account[1]) for account in accounts]
    Phi=np.zeros(shape=(K,I),dtype=float)
    AM=[[mid_to_pos[e[0]] for e in account[1]] for account in accounts ] # Movies of each account
    Time=[array([e[2]%Time_Period for e in account[1]]) for account in accounts ] # Time of the day
    if not Y:
        Y=[random.rand(V,K,Mu) for Mu in M] # Initialize at random the P(v,z|t,i,Lambda^)
        Y_norm=[y.sum(axis=(0,1)) for y in Y] # Compute the normalisation term 
        Y=[divide(y, y_norm) for y,y_norm in zip(Y,Y_norm)] # Normalize
        
    
    for it in range(Nb_it):
        try:
            # One M Step
            # Precompute the expressions we will need multiple times
            Yu_km=[y.sum(axis=0) for y in Y] # Sum over V
            Yuv__=array([y.sum(axis=(1,2)) for y in Y])
            Phi=0*Phi+beta # Add the pseudo count Beta due to Dirichlet prior
            for Y_km,mov in zip(Yu_km,AM): # For each user
                 for j,m in enumerate(mov): # For each of its movies
                    Phi[:,m]+=Y_km[:,j]
            #Normalize
            Phi=Phi/Phi.sum(axis=1)[:,None]
            Theta=[y.sum(axis=2) for y in Y] # Theta without pseudo counts nor normalization
            Theta=array(Theta) # Theta without pseudo counts nor normalization
            Psi=Theta.sum(axis=2)+gamma # Psi with pseudo counts but no normalization
            Theta+=alpha # Theta with the pseudo counts
            # Normalize
            Theta=Theta/(Yuv__+K*alpha)[:,:,None]
            Psi=Psi/Psi.sum(axis=1)[:,None]

            Yuv_m=[y.sum(axis=1) for y in Y]
            Tau=array([(yov_m*tom.transpose()).sum(axis=1) for yov_m,tom in zip(Yuv_m,Time)])
            Tau/=Yuv__ # np.divide by the coefficients

            # Compute Sigma2
            Simple_diff=[Tauov[:,None]-tom[None,:] for Tauov,tom in zip(Tau,Time)]
            Circular_diff2=[array([(diff-Time_Period)**2,(diff+Time_Period)**2,(diff)**2]).min(axis=0) for diff in Simple_diff]
            Sigma2=array([(yov_m*dom).sum(axis=1) for yov_m,dom in zip(Yuv_m,Circular_diff2)])
            Sigma2/=Yuv__
            
            # Deal with singularities
            collapse_threshold=0.1
            # All sigmas which corresponds to collapses
            collapse=(Sigma2<collapse_threshold)
            Sigma2=Sigma2*(1-collapse)+Sigma2.max()*collapse
            Tau=Tau*(1-collapse)+Time_Period*np.random.random(household_size)*collapse
            
            Simple_diff=[Tauov[:,None]-tom[None,:] for Tauov,tom in zip(Tau,Time)]
            Circular_diff2=[array([(diff-Time_Period)**2,(diff+Time_Period)**2,(diff)**2]).min(axis=0) for diff in Simple_diff]
            
            

            # One E step
            Tuv=1/(np.sqrt(2*np.pi*Sigma2))
            Tuvm=[tuv[:,None]*exp(-(diff)/(2*sigma2[:,None])) for diff,tuv,sigma2 in zip(Circular_diff2,Tuv,Sigma2)] # compute the P(t_um|v,Tau,Sigma2)
            PsiTheta=Psi[:,:,None]*Theta
            Y2=[pt[:,:,None]*tuvm[:,None,:] for pt,tuvm in zip(PsiTheta,Tuvm)]
            for u,mov in enumerate(AM):
                for j,m in enumerate(mov):
                    Y[u][:,:,j]=Y2[u][:,:,j]*Phi[None,:,m]

            # Normalize
            Y_norm=[y.sum(axis=(0,1)) for y in Y] # Compute the normalisation term 
            Y=[np.divide(y, y_norm) for y,y_norm in zip(Y,Y_norm)] # Normalize

            
            if verbose:
                # Evaluation
                predictions=dirichlet_prediction(Y,len(composite),accounts,assign=True)
                print performance.stats(composite,{'model':predictions},household_size,cdf=False,measure="similarity")['model']['mean']
                
        except KeyboardInterrupt: # If we are stopped by the user, still return what we can
            predictions=dirichlet_prediction(Y,len(accounts),composite,assign=True)
            perf=performance.stats(composite,{'model':predictions},household_size,cdf=False,measure="similarity")['model']['mean']
            return Y,perf,predictions,{'Tau':Tau,'Phi':Phi,'Sigma2':Sigma2,'Psi':Psi,'Theta':Theta}
    
    predictions=dirichlet_prediction(Y,len(accounts),composite,assign=True)
    perf=performance.stats(composite,{'model':predictions},household_size,cdf=False,measure="similarity")['model']['mean']
    
    return Y,perf,predictions,{'Tau':Tau,'Phi':Phi,'Sigma2':Sigma2,'Psi':Psi,'Theta':Theta}
        

def em_ITVM(accounts,movies,mid_to_pos,composite=None,Y=None,Nb_it=100,verbose=False,alpha=0.001,beta=0.001,gamma=0.001,lambd=1.5,K=50,household_size=2,Time_Period=24.):
    if not composite:
        composite=accounts

    
    # Initialization
    V=household_size
    U=len(accounts) # Number of accounts
    I=len(movies) # Number of movies
    M=[len(account[1]) for account in accounts]
    Phi=np.zeros(shape=(K,I),dtype=float)
    AM=[[mid_to_pos[e[0]] for e in account[1]] for account in accounts ] # Movies of each account
    Time=[array([e[2]*2*pi/(Time_Period) for e in account[1]]) for account in accounts ] # Time in radian
    cosTime=[cos(t) for t in Time]
    sinTime=[sin(t) for t in Time]
    if not Y:
        Y=[random.rand(V,K,Mu) for Mu in M] # Initialize at random the P(v,z|t,i,Lambda^)
        Y_norm=[y.sum(axis=(0,1)) for y in Y] # Compute the normalisation term 
        Y=[divide(y, y_norm) for y,y_norm in zip(Y,Y_norm)] # Normalize

    for it in range(Nb_it):
        try:
            # One M Step
            # Precompute the expressions we will need multiple times
            Yu_km=[y.sum(axis=0) for y in Y] # Sum over V
            Yuv__=array([y.sum(axis=(1,2)) for y in Y])
            Phi=0*Phi+beta # Add the pseudo count Beta due to Dirichlet prior
            for Y_km,mov in zip(Yu_km,AM): # For each user
                 for j,m in enumerate(mov): # For each of its movies
                    Phi[:,m]+=Y_km[:,j]
            #Normalize
            Phi=Phi/Phi.sum(axis=1)[:,None]
            Theta=[y.sum(axis=2) for y in Y] # Theta without pseudo counts nor normalization
            #Psi=[theta.sum(axis=1)+gamma for theta in Theta] # Psi with pseudo counts but no normalization
            Theta=array(Theta) # Theta without pseudo counts nor normalization
            Psi=Theta.sum(axis=2)+gamma # Psi with pseudo counts but no normalization
            Theta+=alpha # Theta with the pseudo counts
            # Normalize
            Theta=Theta/(Yuv__+K*alpha)[:,:,None]
            Psi=Psi/Psi.sum(axis=1)[:,None]

            Yuv_m=[y.sum(axis=1) for y in Y]
            C=array([(ct[None,:]*yv_m).sum(axis=1) for ct,yv_m in zip(cosTime,Yuv_m)])
            S=array([(st[None,:]*yv_m).sum(axis=1) for st,yv_m in zip(sinTime,Yuv_m)])
            Tau=arctan2(S,C)
            R=(sqrt(C*C+S*S)-lambd)/Yuv__ # Distance from the "center of the clock" (radius)
            R=np.maximum(R,0) # Prevent going beyond 0 where the following approximation would not be valid
            
            R2=R*R
            Kappa=R*(2-R2)/(1-R2)


            # One E step
            Tuv=1/iv(0,Kappa)
            Tuvm=[tuv[:,None]*exp(kappa[:,None]*cos(t[None,:]-tau[:,None])) for t,tau,tuv,kappa in zip(Time,Tau,Tuv,Kappa)] # compute the P(t_um|v,Tau,Kappa)
            PsiTheta=Psi[:,:,None]*Theta
            Y2=[pt[:,:,None]*tuvm[:,None,:] for pt,tuvm in zip(PsiTheta,Tuvm)]
            for u,mov in enumerate(AM):
                for j,m in enumerate(mov):
                    Y[u][:,:,j]=Y2[u][:,:,j]*Phi[None,:,m]

            # Normalize
            Y_norm=[y.sum(axis=(0,1)) for y in Y] # Compute the normalisation term 
            Y=[divide(y, y_norm) for y,y_norm in zip(Y,Y_norm)] # Normalize

            
            if verbose:
                # Evaluation
                predictions=dirichlet_prediction(Y,len(composite),accounts,assign=True)
                print performance.stats(composite,{'model':predictions},household_size,cdf=False,measure="similarity")['model']['mean']
                
        except KeyboardInterrupt: # If we are stopped by the user, still return what we can
            predictions=dirichlet_prediction(Y,len(accounts),composite,assign=True)
            perf=performance.stats(composite,{'model':predictions},household_size,cdf=False,measure="similarity")['model']['mean']
            return Y,perf,predictions,{'Tau':Tau,'Phi':Phi,'Kappa':Kappa,'Psi':Psi,'Theta':Theta}
    
    predictions=dirichlet_prediction(Y,len(accounts),composite,assign=True)
    perf=performance.stats(composite,{'model':predictions},household_size,cdf=False,measure="similarity")['model']['mean']
    
    return Y,perf,predictions,{'Tau':Tau,'Phi':Phi,'Kappa':Kappa,'Psi':Psi,'Theta':Theta}


def dirichlet_prediction_time(Y,n_pred,accounts,assign=True):
    Yuv_m=[y.sum(axis=(1,2)) for y in Y]
    predictions=[]
    for yuv_m,account in zip(Yuv_m[0:n_pred],accounts[0:n_pred]):
        clusters=[]
        for e in yuv_m.transpose():
            if assign:
                clusters.append(np.argmax(e))
            else:
                clusters.append(round(e[0]/(e[0]+e[1]),4))
        prediction = (account[0],[(film[0],cluster) for cluster, film in zip(clusters,account[1])])
        predictions.append(prediction)
    return predictions


def em_TAUITM(accounts,movies,mid_to_pos,composite=None,Y=None,L=None,Nb_it=100,verbose=False,alpha=0.001,beta=0.001,gamma=0.001,tau=0.001,rho=0.001,iota=1,K=30,R=10,household_size=2,Genre_weight=0,Time_Discrete=1,roc_psi=False):
    """ Note that there is no em_dir_time_topics as it was the function for the model discretising the time and the pseudo discretization is way better """
    if not composite:
        composite=accounts
    # Initialization
    V=household_size
    U=len(accounts) # Number of accounts
    I=len(movies) # Number of movies
    M=[len(account[1]) for account in accounts]
    Phi=np.zeros(shape=(K,I),dtype=float)
    AM=[[mid_to_pos[e[0]] for e in account[1]] for account in accounts ] # Movies of each account
    Time=[[e[2]/float(Time_Discrete) for e in account[1]] for account in accounts ] # Times of Week of each account
    T=int(ceil(max(map(max,Time)))) # Max Time
    Gamma=np.zeros(shape=(R,T),dtype=float)

    # Represent all the genres as integers
    if Genre_weight:
        all_genres=set()
        all_genres.add(UNKNOWN)
        for mid,movie in movies.iteritems():
            for genre in movie[1]:
                if genre!='':
                    all_genres.add(genre)
        
        all_genres=list(all_genres)
        genre_to_gid={genre:gid for gid,genre in enumerate(all_genres)}
        genre_to_gid[u'']=genre_to_gid[UNKNOWN] # Handle the unknown case
        Genre=[[genre_to_gid[genre] for genre in movie[1]] for mid,movie in movies.iteritems()] 
        #Genre=[[genre_to_gid[genre] for genre in movie[1]] if not(movie[0] in known_movies) else [] for mid,movie in movies.iteritems()] 
        C=len(all_genres) # Number of genres
        G=[len(genres) for genres in Genre]
        W=zeros(shape=(K,C))
    else:
		W=None

    if not Y:
        Y=[random.rand(V,K,R,Mu) for Mu in M] # Initialize at random the P(v,z|t,i,Lambda^)
        Y_norm=[y.sum(axis=(0,1,2)) for y in Y] # Compute the normalisation term 
        Y=[divide(y, y_norm) for y,y_norm in zip(Y,Y_norm)] # Normalize

    if not L:
		if Genre_weight:
			L=[random.rand(K,Gi) for Gi in G] # Initialize P(z|i,g)
			L=[Lozg/Lozg.sum(axis=0)[None,:] for Lozg in L]
		else:
			L=None
    
    for it in range(Nb_it):
        try:
            # One M Step

            # Precompute what we will need multiple times
            Yu_krm=[Yovkrm.sum(axis=0) for Yovkrm in Y] # P(z,r|u,m,Lambda^)
            Yu__rm=[Yo_krm.sum(axis=0) for Yo_krm in Yu_krm] # P(r|u,m,Lambda^)
            Yu_k_m=[Yo_krm.sum(axis=1) for Yo_krm in Yu_krm] # P(z|u,m,Lambda^)
            del Yu_krm
            Yuvkr_=array([Yovkrm.sum(axis=3) for Yovkrm in Y]) # Sum_{m}^Mu P(z,r|u,m,Lambda^) for each u=1...U


            # Compute Phi
            if Genre_weight:
                Phi=Genre_weight*array([Lozg.sum(axis=1) for Lozg in L]).transpose()+beta # Account for the Genre contribution and smoothing
            else:
                Phi=0*Phi+beta
                
            for Yo_k_m,mov in zip(Yu_k_m,AM):
                for j,i in enumerate(mov):
                    Phi[:,i]+=Yo_k_m[:,j]
            Phi=Phi/Phi.sum(axis=1)[:,None] # Normalize

            # Compute Gamma
            Gamma=0*Gamma+tau # Pseudo count
            for Yo__rm,times in zip(Yu__rm,Time):
                for j,t in enumerate(times):
                    floor_t=floor(t)
                    ceil_t=ceil(t)
                    if floor_t==ceil_t: # Handle the particular case where we have an integer
                        ceil_t+=1
                    prop=t-floor_t
                    Gamma[:,int(floor_t)]+=(1-prop)*Yo__rm[:,j]
                    Gamma[:,int(ceil_t)%T]+=(prop)*Yo__rm[:,j]
            Gamma=Gamma/Gamma.sum(axis=1)[:,None] # Normalize

            del Yu__rm
            
            if Genre_weight:
                # Compute W
                W=0*W+iota # Pseudo count
                for Yozg,genres in zip(L,Genre):
                    for j,g in enumerate(genres):
                        W[:,g]+=Yozg[:,j]
                W/=W.sum(axis=1)[:,None]


            # Start to compute Theta, but compute Psi in between (for speed, as we can use Theta without pseudo counts nor normalization to compute Psi)
            Theta=Yuvkr_.sum(axis=3) # Theta without pseudo count nor normalization
            Psi=Theta.sum(axis=2)+gamma # Psi with pseudo counts
            Psi/=Psi.sum(axis=1)[:,None]
            Theta+=alpha # Add the pseudo counts to Theta
            Theta/=Theta.sum(axis=2)[:,:,None] # Normalize

            # Compute Pi
            Pi=Yuvkr_.sum(axis=2)+rho # Pi with pseudo count but normalization
            Pi/=Pi.sum(axis=2)[:,:,None]


            # One E step

            PsiPi=Psi[:,:,None]*Pi
            PsiPiTheta=PsiPi[:,:,None,:]*Theta[:,:,:,None]
            del PsiPi

            # Estimate 
            u=0
            for mov,times in zip(AM,Time):
                j=0
                for i,t in zip(mov,times):
                    floor_t=floor(t)
                    ceil_t=ceil(t)
                    if floor_t==ceil_t: # Handle the particular case where we have an integer
                        ceil_t+=1
                    prop=t-floor_t
                    Y[u][:,:,:,j]=PsiPiTheta[u]*Phi[None,:,None,i]*((1-prop)*Gamma[None,None,:,int(floor_t)]+prop*Gamma[None,None,:,int(ceil_t)%T])
                    j+=1 
                u+=1
            Y_norm=[y.sum(axis=(0,1,2)) for y in Y] # Compute the normalisation term 
            Y=[divide(y, y_norm) for y,y_norm in zip(Y,Y_norm)] # Normalize

            del PsiPiTheta
            
            if Genre_weight:
                # Estimate L
                for i,genres in enumerate(Genre):
                    for j,g in enumerate(genres):
                        L[i][:,j]=Phi[:,i]*W[:,g]
                L=[Lozg/Lozg.sum(axis=0)[None,:] for Lozg in L]

            if verbose:
                if household_size>1:
                    # Evaluation
                    predictions=dirichlet_prediction_time(Y,len(composite),accounts,assign=True)
                    print it,performance.stats(composite,{'model':predictions},household_size,cdf=False,measure="similarity",)['model']['mean']
                else:
                    print it
            if roc_psi:
                print 'roc auc:',roc_auc_score(array(nb_users(composite))-1,Psi[0:len(composite)].min(axis=1))
            
        except KeyboardInterrupt: # If the algorithm is stopped still return what you can
            if household_size>1:
                predictions=dirichlet_prediction_time(Y,len(accounts),composite,assign=True)
                perf=performance.stats(composite,{'model':predictions},household_size,cdf=False,measure="similarity")['model']['mean']
                return Y,perf,predictions,{'R':R,'K':K,'Time_Discrete':Time_Discrete,'T':T,'Pi':Pi,'Phi':Phi,'Psi':Psi,'Theta':Theta,'Gamma':Gamma,'W':W}
            else:
                return Y,{'R':R,'K':K,'Time_Discrete':Time_Discrete,'T':T,'Pi':Pi,'Phi':Phi,'Psi':Psi,'Theta':Theta,'Gamma':Gamma,'W':W}
            
            
            
    if household_size>1:
        predictions=dirichlet_prediction_time(Y,len(accounts),composite,assign=True)
        perf=performance.stats(composite,{'model':predictions},household_size,cdf=False,measure="similarity")['model']['mean']
        return Y,perf,predictions,{'R':R,'K':K,'Time_Discrete':Time_Discrete,'T':T,'Pi':Pi,'Phi':Phi,'Psi':Psi,'Theta':Theta,'Gamma':Gamma,'W':W}
    else:
        return Y,{'R':R,'K':K,'Time_Discrete':Time_Discrete,'T':T,'Pi':Pi,'Phi':Phi,'Psi':Psi,'Theta':Theta,'Gamma':Gamma,'W':W}

def dirichlet_prediction_time_param(params,n_pred,accounts,mid_to_pos,assign=True):
    """ Predict the active users from the params (and not the Y) """
    predictions=[]
    R,K,Time_Discrete,T,Pi,Phi,Psi,Theta,Gamma=params['R'],params['K'],params['Time_Discrete'],params['T'],params['Pi'],params['Phi'],params['Psi'],params['Theta'],params['Gamma']
    for u,account in enumerate(accounts[0:n_pred]):
        times,mov,Vu,Mu=([e[2]/float(Time_Discrete) for e in account[1]],[mid_to_pos[e[0]] for e in account[1]],len(account[0]),len(account[1]))
        Yovkrm=zeros(shape=(Vu,K,R,Mu))
        PsiPi=Psi[u][:,None]*Pi[u]
        PsiPiTheta=PsiPi[:,None,:]*Theta[u][:,:,None]
        for j,i,t in zip(range(Mu),mov,times):
            floor_t=floor(t)
            ceil_t=ceil(t)
            if floor_t==ceil_t: # Handle the particular case where we have an integer
                ceil_t+=1
            prop=t-floor_t
            Yovkrm[:,:,:,j]=PsiPiTheta*Phi[None,:,None,i]*((1-prop)*Gamma[None,None,:,int(floor_t)]+prop*Gamma[None,None,:,int(ceil_t)%T])
        Yovkrm/=Yovkrm.sum(axis=(0,1,2))
        yov__m=Yovkrm.sum(axis=(1,2))
        clusters=[]
        for e in yov__m.transpose():
            if assign:
                clusters.append(np.argmax(e))
            else:
                clusters.append(round(e[1]/(e[0]+e[1]),4))
        prediction = (account[0],[(film[0],cluster) for cluster, film in zip(clusters,account[1])])
        predictions.append(prediction)
    return predictions


def signal_handling(arg1,arg2):
    global terminate
    terminate=True


def em_TAUITM_memory(accounts,movies,mid_to_pos,composite=None,params=None,Nb_it=100,verbose=True,alpha=0.001,beta=0.001,gamma=0.001,tau=0.001,rho=0.001,iota=1,K=30,R=10,household_size=2,Genre_weight=0.0,Time_Discrete=1,roc_psi=False,assign=True,report_ident=True):
    """ Memory efficient EM version for CAMARA.
    WARNING: Assume the number of users known!
    Special weighting for genres """
    # Initialization
    UNKNOWN='Unknown'
    V=[len(account[0]) for account in accounts]
    U=len(accounts) # Number of accounts
    I=len(movies) # Number of movies
    M=[len(account[1]) for account in accounts]

    AM=[[mid_to_pos[e[0]] for e in account[1]] for account in accounts ] # Movies of each account
    Time=[[e[2]/float(Time_Discrete) for e in account[1]] for account in accounts ] # Times of Week of each account
    T=int(ceil(max(map(max,Time)))) # Max Time


    # Represent all the genres as integers
    if Genre_weight:
        all_genres=set()
        all_genres.add(UNKNOWN)
        for mid,movie in movies.iteritems():
            for genre in movie[1]:
                if genre!='':
                    all_genres.add(genre)
    
        all_genres=list(all_genres)
        genre_to_gid={genre:gid for gid,genre in enumerate(all_genres)}
        genre_to_gid[u'']=genre_to_gid[UNKNOWN] # Handle the unknown case
        Genre=[[genre_to_gid[genre] for genre in movie[1]] for mid,movie in movies.iteritems()] 
        C=len(all_genres) # Number of genres
        G=[len(genres) for genres in Genre]


    if params:
        starting_it=1
        Nb_it+=1
    else:
        starting_it=0

    if not params:
        # Create the global parameters
        Phi=beta*ones(shape=(K,I),dtype=float)
        Gamma=tau*ones(shape=(R,T),dtype=float)
        if Genre_weight:
            W=iota*ones(shape=(K,C))
        else:
            W=None

        # Create the local ones
        Psi=[zeros(shape=Vu) for Vu in V]
        Theta=[zeros(shape=(Vu,K)) for Vu in V]
        Pi=[zeros(shape=(Vu,R)) for Vu in V]
    else:
        W,Pi,Phi,Psi,Theta,Gamma=params['W'],params['Pi'],params['Phi'],params['Psi'],params['Theta'],params['Gamma']

    global terminate # Variable to handle gracefully finishing the current iteration after a SIGINT
    terminate=False
    signal.signal(signal.SIGINT,signal_handling)
    for it in range(starting_it,Nb_it):
            # Make a copy which will be used for estimation
            GammaOld=Gamma.copy()
            PhiOld=Phi.copy()
            if Genre_weight:
                WOld=W.copy()

            # Reset the parameters to be maximized
            Phi=beta*ones(shape=(K,I),dtype=float)
            Gamma=tau*ones(shape=(R,T),dtype=float)
            if Genre_weight:
                W=iota*ones(shape=(K,C))

            # For each account
            for u,times,mov,Vu,Mu in zip(range(U),Time,AM,V,M):
                if not it: # At first initialize at random
                    Yovkrm=random.rand(Vu,K,R,Mu)
                else:
                    Yovkrm=zeros(shape=(Vu,K,R,Mu))
                    PsiPi=Psi[u][:,None]*Pi[u]
                    PsiPiTheta=PsiPi[:,None,:]*Theta[u][:,:,None]
                    for j,i,t in zip(range(Mu),mov,times):
                        floor_t=floor(t)
                        ceil_t=ceil(t)
                        if floor_t==ceil_t: # Handle the particular case where we have an integer
                            ceil_t+=1
                        prop=t-floor_t
                        Yovkrm[:,:,:,j]=PsiPiTheta*PhiOld[None,:,None,i]*((1-prop)*GammaOld[None,None,:,int(floor_t)]+prop*GammaOld[None,None,:,int(ceil_t)%T])
                Yovkrm/=Yovkrm.sum(axis=(0,1,2))

                # Update the local parameters
                Theta[u]=Yovkrm.sum(axis=(2,3)) # Theta without pseudo count nor normalization
                Psi[u]=Theta[u].sum(axis=1)+gamma
                Psi[u]/=Psi[u].sum()
                Theta[u]+=alpha
                Theta[u]/=Theta[u].sum(axis=1)[:,None]

                Pi[u]=Yovkrm.sum(axis=(1,3))+rho
                Pi[u]/=Pi[u].sum(axis=1)[:,None]

                # Add the contribution to the global parameters
                Yo_k_m=Yovkrm.sum(axis=(0,2))
                for j,i in enumerate(mov):
                    Phi[:,i]+=Yo_k_m[:,j]

                Yo__rm=Yovkrm.sum(axis=(0,1))
                for j,t in enumerate(times):
                    floor_t=floor(t)
                    ceil_t=ceil(t)
                    if floor_t==ceil_t: # Handle the particular case where we have an integer
                        ceil_t+=1
                    prop=t-floor_t
                    Gamma[:,int(floor_t)]+=(1-prop)*Yo__rm[:,j]
                    Gamma[:,int(ceil_t)%T]+=(prop)*Yo__rm[:,j]
            
            if Genre_weight:
            # For each movie
                for i,Gi,genres in zip(range(I),G,Genre):
                    Lozj=zeros(shape=(K,Gi))
                    if not it: 
                        Lozj=random.rand(K,Gi)
                    else:
                        for j,g in enumerate(genres):
                            Lozj[:,j]=PhiOld[:,i]*WOld[:,g]
                    Lozj/=Lozj.sum(axis=0)


                    # Add the contribution to the global parameters
                    Phi[:,i]+=Genre_weight*Lozj.sum(axis=1)/len(genres)

                    for j,g in enumerate(genres):
                        W[:,g]+=Lozj[:,j]


            # Normalize the global parameters
            Gamma=Gamma/Gamma.sum(axis=1)[:,None]
            Phi=Phi/Phi.sum(axis=1)[:,None] # Normalize
            if Genre_weight:
                W/=W.sum(axis=1)[:,None]


            if verbose:
                params={'R':R,'K':K,'beta':beta,'iota':iota,'gamma':gamma,'alpha':alpha,'Time_Discrete':Time_Discrete,'T':T,'Pi':Pi,'Phi':Phi,'Psi':Psi,'Theta':Theta,'Gamma':Gamma,'W':W}
                
                if (report_ident):
                    predictions=dirichlet_prediction_time_param(params,len(composite),accounts,mid_to_pos,assign=assign)
                    perf=performance.stats(composite,{'model':predictions},household_size,cdf=False,measure="similarity")['model']['mean']
                else:
                    perf=None
                    
                print it,perf
            
            if terminate:
                terminate=False
                break

    params={'R':R,'K':K,'beta':beta,'iota':iota,'gamma':gamma,'alpha':alpha,'Time_Discrete':Time_Discrete,'T':T,'Pi':Pi,'Phi':Phi,'Psi':Psi,'Theta':Theta,'Gamma':Gamma,'W':W}
    if household_size>1:
        
        if report_ident: 
            predictions=dirichlet_prediction_time_param(params,len(composite),accounts,mid_to_pos,assign=assign)
            perf=performance.stats(composite,{'model':predictions},household_size,cdf=False,measure="similarity")['model']['mean']
        else:
            predictions=dirichlet_prediction_time_param(params,len(composite),accounts,mid_to_pos,assign=assign)
            perf=None,None
        return perf,predictions,params
    else:
        return params


