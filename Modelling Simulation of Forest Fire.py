#from asyncio.windows_events import NULL
import matplotlib.pyplot as plt
from random import *
from timeit import repeat
import numpy as np
from copy import deepcopy
from matplotlib import animation
import time
#from asyncio.windows_events import NULL
from numba import jit, prange
from datetime import datetime

# probability that the site has a tree
probTree = 0.8  

# probability the tree is burning
probBurning = 0.01 

# probability  the tree is immune to burning
probImmune = 0.3  

# probability of a lightning strike on the tree
probLightning = 0.001  

# No tree found on the sites
EMPTY = 0  

# Tree is not burning
NONBURNING = 1  

# Tree is burning
BURNING = 2  

# ForestGrid boarder
BORDER = 3

# Parallel Simulation
UseParallelSim = False

def createforestgrid(n):

    forestgrid = BORDER * np.ones((n+2, n+2))
    for i in range(1, n+1):
        for j in range(1, n+1):
            if random() < probTree:  
                if random() < probBurning or (i==1 and j==1): 
                     forestgrid[i, j] = BURNING  
                else:
                    
                    forestgrid[i, j] = NONBURNING
            else:
                forestgrid[i, j] = EMPTY  

    return forestgrid


@jit(forceobj = True, parallel = True)
def createforestgridparallelmethod(n):

    forestgrid = BORDER * np.ones((n+2, n+2))
    for i in prange(1, n+1):
        for j in prange(1, n+1):
            if random() < probTree:  
                if random() < probBurning or (i==1 and j==1): 
                    forestgrid[i, j] = BURNING  
                else:
                
                    forestgrid[i, j] = NONBURNING
            else:
                forestgrid[i, j] = EMPTY  

    return forestgrid


def createfirespread(ground, neighbourhood):
    value = 0
    if ground == EMPTY or ground == BURNING:
        value = EMPTY
    else:
        if random() < probLightning:
            value = BURNING
        elif BURNING in neighbourhood:
            if random() < probImmune:
                value = NONBURNING
            else:
                value = BURNING
        else:
            value = NONBURNING
    return value
            


def applyfirespread(ground, neighbourhood):
    rows = ground.shape[0] - 2
    columns = ground.shape[1] - 2
    saveGroundSite = deepcopy(ground)

    for i in range(1, rows+1):
        for j in range(1, columns+1):
            siteGround = ground[i, j]
            N = ground[i-1, j]
            NE = ground[i-1, j+1]
            E = ground[i, j + 1]
            SE = ground[i + 1, j + 1]
            S = ground[i + 1, j]
            SW = ground[i + 1, j - 1]
            W = ground[i, j - 1]
            NW = ground[i - 1, j - 1]
            if neighbourhood == 4:
                neighbourhoodtype = [N, E, S, W]
                saveGroundSite[i, j] = createfirespread(siteGround, neighbourhoodtype)
            else:
                neighbourhoodtype = [N, NE, E, SE, S, SW, W, NW]
                saveGroundSite[i, j] = createfirespread(siteGround, neighbourhoodtype)
            
            
    return saveGroundSite

@jit(forceobj=True,parallel=True)
def applyfirespreadparallel(ground, neighbourhood):
    rows = ground.shape[0] - 2
    columns = ground.shape[1] - 2
    saveGroundSite = deepcopy(ground)
    

    for i in prange(1, rows+1):
        for j in prange(1, columns+1):
            siteGround = ground[i, j]
            N = ground[i-1, j]
            NE = ground[i-1, j+1]
            E = ground[i, j + 1]
            SE = ground[i + 1, j + 1]
            S = ground[i + 1, j]
            SW = ground[i + 1, j - 1]
            W = ground[i, j - 1]
            NW = ground[i - 1, j - 1]
            if neighbourhood == 4:
                surroundings = [N, E, S, W]
                saveGroundSite[i, j] = createfirespread(siteGround, surroundings)
            else:
                surroundings = [N, NE, E, SE, S, SW, W, NW]
                saveGroundSite[i, j] = createfirespread(siteGround, surroundings)
            
    return saveGroundSite


def startfiresimulation(groundSize,time,parallelSim,neighbourhoodtype):
    forestGrounds = [] 
    if parallelSim == False:
        forestGround = createforestgrid(groundSize)
    else:
        forestGround = createforestgridparallelmethod(groundSize)
        
        
    for timeStep in range(time):
        if parallelSim == False:
            forestGround = applyfirespread(forestGround,neighbourhoodtype)
        else:
            forestGround = applyfirespreadparallel(forestGround,neighbourhoodtype)
           
        forestGrounds.append(forestGround)
    
    return forestGrounds




      
start = time.time()
runforestfiresimulation = startfiresimulation(1200,20,parallelSim=False,neighbourhoodtype=4)
end = time.time()
print(end-start)


fig,ax = plt.subplots()
ims = []


for i in range(len(runforestfiresimulation)):
    im = ax.imshow(runforestfiresimulation[i],cmap='cool', animated=True)
    if i == 0:
        ax.imshow(runforestfiresimulation[i])  
    ims.append([im])


ani = animation.ArtistAnimation(fig, ims,  interval=100, blit=True,repeat=True)
plt.show()