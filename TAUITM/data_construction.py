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

import unicodecsv as csv
from collections import defaultdict
import random
import numpy as np
import time as libtime

# Define the constants about time
HOUR_IN_DAY=24
MIN_IN_HOUR=60
SEC_IN_MIN=60
DAY_IN_WEEK=7
WEEK_IN_YEAR=52
SEC_IN_HOUR=MIN_IN_HOUR*SEC_IN_MIN
SEC_IN_DAY=HOUR_IN_DAY*SEC_IN_HOUR
SEC_IN_WEEK=SEC_IN_DAY*DAY_IN_WEEK
SEC_IN_YEAR=WEEK_IN_YEAR*SEC_IN_WEEK

def read_movies(filename):
    """ Read all the movies from a csv file and get word tokens"""
    with open(filename,"r") as f:
        csvreader = csv.reader(f,encoding='utf-8')
        next(csvreader, None)  # skip the headers
        movies={}
        for row in csvreader:  
            row_split=row[0].split('::')
            if len(row_split)>2:
                mid,title,genres=row[0].split('::')
                # Get the information and also tokenize the text
                movies[int(mid)]=[title,genres.split('|'),True,True,True]
            else:
                mid,title=row[0].split('::')
                # Get the information and also tokenize the text
                movies[int(mid)]=[title,[],True,True,True]
            
    names = [movie[1][0] for movie in movies.items()]
    mids = [k for k,movie in movies.iteritems()] # id of movies
    mid_to_pos={} # map id of movies to position in the vectors/arrays
    for i,mid in enumerate(mids):
        mid_to_pos[mid]=i

    return movies,names,mid_to_pos




def read_ratings_MT(filename):
    with open(filename, "r") as f:
        ratings=defaultdict(list)
        for line in f:
            uid,mid,r,t=line.rstrip('\n').split('::')
            ratings[int(uid)].append([int(mid),int(r),int(t)])
    return ratings


def get_user_rating_min(ratings,n=20):
    """Return the keys of users with more than n ratings"""
    return [key for key,value in ratings.iteritems() if len(value)>n]

def get_user_rating_max(ratings,n=20):
    """Return the keys of users with at most ratings"""
    return [key for key,value in ratings.iteritems() if len(value)<=n]


def generate_composite_users(ratings,movies,household_size_distribution=[0,1],min_ratings=20,remove_unknown=False,close=False,big=True,time=None,seed=None,dev=0.,test=0.,min_time_distance=None):
    """ Generate composite users of size houshold_size from users having at least min_rating
    Use each user once
    min_time_distance: min distance in second from the previous rating for a rating to be considered
    """
    if seed!=None: # Possibility to use a seed in order to get a deterministic result
        np.random.seed(seed)
    if isinstance( household_size_distribution, int ): # The household size distribution is only an int
        household_size_distribution=np.zeros(household_size_distribution)
        household_size_distribution[-1]=1
    l_household_size_distribution=len(household_size_distribution)
    # Remove all rating on unknown films (bug of the API, synopsis not available)
    if remove_unknown:
        r2={}
        for user, rating in ratings.iteritems():
            r2[user]=[r for r in rating if (movies.get(r[0]) or [None,None,None,''])[3]]
        ratings=r2
    if min_time_distance: # remove items rated just after another
        r2={}
        for user, rating in ratings.iteritems():
            rating=sorted(rating, key=lambda r: r[2]) # Sort the ratings by time
            new_rating=[]
            previous_time=-float("inf")
            for r in rating:
                if r[2]>=previous_time+min_time_distance: # The distance is respected so add the rating
                    new_rating.append(r)
                previous_time=r[2]
            r2[user]=new_rating
        ratings=r2
            
        
    
    if big:
        users=get_user_rating_min(ratings,min_ratings) # Pool of users which can be used for generation
    else:
        users=get_user_rating_max(ratings,min_ratings)
        
    # Remove the users without any rating
 
    i=0
    while i<len(users):
        if len(ratings[users[i]])==0:
            del users[i]
        else:
            i+=1
        
    if close: # Will put to the same account users that rated movies in the same period
            users=sorted(users,key= lambda user: np.max([int(r[2]/SEC_IN_WEEK) for r in ratings[user]]))
            #users=sorted(users,key= lambda user: np.mean([r[2] for r in ratings[user]]))
    else:# Shuffle
        random.shuffle(users)
        
    if not test+dev:
        composite_ratings=[]
    else:
        composite_ratings_train=[]
        composite_ratings_dev=[]
        composite_ratings_test=[]
        
    nb_users=len(users)
    i=0
    while i < nb_users:
        household_size=1+min([np.random.choice(l_household_size_distribution, 1, p=household_size_distribution)[0],nb_users-i-1]) # Draw the household size according to the distribution
        users_id=[]
        users_ratings=[]
        if test+dev: # If we also build a test set, we need to keep track of the absolute time
            absolute_time=[]
        for j in range(household_size):
            users_id.append(users[i])
            r=ratings[users[i]]
            
            for e in r:
                if not time:
                    users_ratings.append((e[0],j))
                else: # Handle the conversion in hour of the day/week
                    t=libtime.gmtime(e[2])
                    if time=='day':
                        t=t.tm_hour+t.tm_min/float(MIN_IN_HOUR)+t.tm_sec/float(SEC_IN_HOUR)
                    elif time=='week':
                        t=(t.tm_wday)*HOUR_IN_DAY+t.tm_hour+t.tm_min/float(MIN_IN_HOUR)+t.tm_sec/float(SEC_IN_HOUR)
                    elif time=='all':
                        t=str(t.tm_year)+' '+str(t.tm_mon)+' '+str(t.tm_mday)+' '+str(t.tm_hour)+' '+str(t.tm_min)+' '+str(t.tm_wday)
                    elif time=='absolute':
                        t=e[2]/float(SEC_IN_HOUR)
                    users_ratings.append((e[0],j,t))
                    if test+dev:
                        absolute_time.append(e[2]) # Absolute Timestamp
            i+=1 # Go to the next user
        if not test+dev: # If we don't build any test set, we take all events
            composite_ratings.append((users_id,users_ratings))
        else: # We give the first events to the trainset and the last ones to the test set
            nb_events=len(absolute_time)
            if isinstance( test, float ):
                nb_test=int(nb_events*test)
            elif isinstance( test, int ):
                nb_test=test
            else:
                raise TypeError("Variable test should be a float or an int")
                
            if isinstance( test, float ):
                nb_dev=int(nb_events*dev)
            elif isinstance( test, int ):
                nb_dev=dev
            else:
                raise TypeError("Variable dev should be a float or an int")
                

            users_ratings_train=[]
            users_ratings_dev=[]
            users_ratings_test=[]
            pos=0
            for t,event in sorted(list(zip(absolute_time,users_ratings))): # Sort the events by absolute time stamps
                if pos<nb_events-nb_test-nb_dev:
                    users_ratings_train.append(event)
                elif pos<nb_events-nb_test:
                    users_ratings_dev.append(event)
                else:
                    users_ratings_test.append(event)
                pos+=1
            composite_ratings_train.append((users_id,users_ratings_train))
            composite_ratings_dev.append((users_id,users_ratings_dev))
            composite_ratings_test.append((users_id,users_ratings_test))
    if not test+dev:            
        return composite_ratings
    else:
        return composite_ratings_train,composite_ratings_dev,composite_ratings_test




