import numpy as np
import cupy as cp
import math
import sys

# Parameters for how to run the simulation
# The program runs rampingTurnsStart-exclusive and rampingTurnsGoal-inclusive
# Note: Increasing trackingTurns gives more conclusive results, but also lengthens the runtime significantly.
trackingTurns = int(sys.argv[1])
rampingTurnsStart = int(sys.argv[2])
rampingTurnsGoal = int(sys.argv[3])
rampingInterval = int(sys.argv[4])
totalParticles = 100000

# Data storage array initialization
emittanceX = np.zeros(trackingTurns)
emittanceY = np.zeros(trackingTurns)
emittanceZ = np.zeros(trackingTurns)
averageEmittanceX = np.zeros(int((rampingTurnsGoal-rampingTurnsStart) / rampingInterval))
averageEmittanceY = np.zeros(int((rampingTurnsGoal-rampingTurnsStart) / rampingInterval))
averageEmittanceZ = np.zeros(int((rampingTurnsGoal-rampingTurnsStart) / rampingInterval))

# Constants
C = 299792458
Q = np.array([0.228, 0.21, 0.01])
totalMu = Q * (math.pi) * 2
epsilon = np.array([11.3e-9, 1e-9, 0.06 * 6.6e-4])
IPBeta = np.array([0.8, 0.072, 0.06 / 6.6e-4])
crabCavityBeta = np.array([1300, 30, 0.06 / 6.6e-4])
IPAlpha = np.array([0, 0, 0])
crabCavityAlpha = np.array([0, 0, 0])
IPtoCCAMu = np.array([math.radians(88), 0, 0])
CCBtoIPMu = np.array([math.radians(87), 0, 0])
CCAtoCCBMu = totalMu - (IPtoCCAMu + CCBtoIPMu)
kc = 2 * math.pi * 200e6 / C
phiC = 0
thetaC = 12.5e-3

# Lambda Initilization
lambdaXGoal = thetaC / np.sqrt(IPBeta[0] * crabCavityBeta[0])
lambdaXAGoal = ((2*thetaC*math.sin(CCBtoIPMu[0] - math.pi * Q[0])*math.sin(math.pi * Q[0]))/(math.sin(CCBtoIPMu[0] + IPtoCCAMu[0] - 2*math.pi*Q[0]))) / np.sqrt(IPBeta[0] * crabCavityBeta[0])
lambdaXBGoal = ((2*thetaC*math.sin(IPtoCCAMu[0] - math.pi * Q[0])*math.sin(math.pi * Q[0]))/(math.sin(CCBtoIPMu[0] + IPtoCCAMu[0] - 2*math.pi*Q[0]))) / np.sqrt(IPBeta[0] * crabCavityBeta[0])
lambdaYAGoal = 0
lambdaYBGoal = 0

# Boost and Inverse Boost matrices
L = cp.array([[1, 0, 0, 0, thetaC, 0],
              [0, 1, 0, 0, 0, 0],
              [0, 0, 1, 0, 0, 0],
              [0, 0, 0, 1, 0, 0],
              [0, 0, 0, 0, 1, 0],
              [0, -thetaC, 0, 0, 0, 1]])
inverseL = cp.array([[1, 0, 0, 0, -thetaC, 0],
                     [0, 1, 0, 0, 0, 0],
                     [0, 0, 1, 0, 0, 0],
                     [0, 0, 0, 1, 0, 0],
                     [0, 0, 0, 0, 1, 0],
                     [0, thetaC, 0, 0, 0, 1]])

# Crab Cavity Kick Functions
crabCavityPXKick = cp.ElementwiseKernel(
  "float64 pPX, float64 pZ, float64 pLambdaX, float64 pKC, float64 pPhiC",
  "float64 rPX",
  "rPX = pPX - (pLambdaX*sin(pKC*pZ+pPhiC)) / pKC",
  "crabCavityPXKick")

crabCavityPYKick = cp.ElementwiseKernel(
  "float64 pPY, float64 pZ, float64 pLambdaY, float64 pKC, float64 pPhiC",
  "float64 rPY",
  "rPY = pPY - (pLambdaY*sin(pKC*pZ+pPhiC)) / pKC",
  "crabCavityPYKick")

crabCavityDeltaKick = cp.ElementwiseKernel(
  "float64 pDelta, float64 pX, float64 pY, float64 pZ, float64 pLambdaX, float64 pLambdaY, float64 pKC, float64 pPhiC",
  "float64 rDelta",
  "rDelta = pDelta - (pLambdaX*pX + pLambdaY*pY)*cos(pKC*pZ+pPhiC)",
  "crabCavityDeltaKick")

# Initialize Particles
# [input] pBeta: Beta of intersection/interaction point
# [input] pAlpha: Alpha of intersection/interaction point
# [input] pEpsilon: epsilon
# [input] particleAmount: Number of particles to create
# [return] rCollection: A (particleAmount x 6) array, with each column representing a particle and each row representing
# x, pX, y, pY, z, delta
def get(pBeta, pAlpha, pEpsilon, particleAmount):
  dummyCollection = np.empty((6, particleAmount))
  for i in range(3):
    sigma = math.sqrt(pBeta[i] * pEpsilon[i])
    u = cp.random.normal(size=(1, particleAmount)).get()
    v = cp.random.normal(size=(1, particleAmount)).get()
    dummyCollection[i * 2] = sigma * u
    dummyCollection[i * 2 + 1] = (sigma * (v - pAlpha[i] * u)) / pBeta[i]
  rCollection = cp.asarray(dummyCollection)
  return rCollection

# Create transport map between two locations
# [input] pBeta1, pBeta2: The betas of the start and end point, respectively
# [input] pAlpha1, pAlpha2: The alphas of the start and end point, respectively
# [input] pDeltaMu: The change in mu between the start and end point
# [return] rCmap: The transport map
def transportMap(pBeta1, pBeta2, pAlpha1, pAlpha2, pDeltaMu):
  dummyMap = np.zeros((6, 6))
  for i in range(3):
    dummyMap[i * 2, i * 2] = math.sqrt(pBeta2[i] / pBeta1[i]) * (
        math.cos(pDeltaMu[i]) + pAlpha1[i] * math.sin(pDeltaMu[i]))
    dummyMap[i * 2, i * 2 + 1] = math.sqrt(pBeta1[i] * pBeta2[i]) * math.sin(pDeltaMu[i])
    dummyMap[i * 2 + 1, i * 2] = -(
        (1 + pAlpha1[i] * pAlpha2[i]) * math.sin(pDeltaMu[i]) + (pAlpha2[i] - pAlpha1[i]) * math.cos(
      pDeltaMu[i])) / math.sqrt(pBeta1[i] * pBeta2[i])
    dummyMap[i * 2 + 1, i * 2 + 1] = math.sqrt(pBeta1[i] / pBeta2[i]) * (
        math.cos(pDeltaMu[i]) - pAlpha2[i] * math.sin(pDeltaMu[i]))
  rCmap = cp.asarray(dummyMap)
  return rCmap

# Runs the crab cavity kick on all three directions
# [input] pCollection: The particle array to kick
# [input] pLambdaX: The strength of the crab cavity in the X direction
# [input] pLambdaY: The strength of the crab cavity in the Y direction
# [input] pKC: kc
# [input] pPhiC: phiC
def crabCavityKick(pCollection, pLambdaX, pLambdaY, pKC, pPhiC):
  pCollection[1, :] = crabCavityPXKick(pCollection[1], pCollection[4], pLambdaX, pKC, pPhiC)
  pCollection[3, :] = crabCavityPYKick(pCollection[3], pCollection[4], pLambdaY, pKC, pPhiC)
  pCollection[5, :] = crabCavityDeltaKick(pCollection[5], pCollection[0], pCollection[2], pCollection[4], pLambdaX,
                                          pLambdaY, pKC, pPhiC)

# Does a cycle around the storage ring with two crab cavities and boosting
#  - Used for tracking at full crab cavity strength
# [input] pCollection: The particle array to track
# [input] pIPtoCCAMap: The transport map from the IP to Crab Cavity A
# [input] pCCAtoCCBMap: The transport map from Crab Cavity A to Crab Cavity B
# [input] pCCBtoIPMap: The transport map from Crab Cavity B to the IP
# [input] pLambdaXA: The strength of Crab Cavity A in the X direction
# [input] pLambdaYA: The strength of Crab Cavity A in the Y direction
# [input] pLambdaXB: The strength of Crab Cavity B in the X direction
# [input] pLambdaYB: The strength of Crab Cavity B in the Y direction
def twoCrabTrack(pCollection, pIPtoCCAMap, pCCAtoCCBMap, pCCBtoIPMap, pLambdaXA, pLambdaYA, pLambdaXB, pLambdaYB):
  pCollection[:] = cp.matmul(pIPtoCCAMap, cp.matmul(inverseL, pCollection))
  crabCavityKick(pCollection, pLambdaXA, pLambdaYA, kc, phiC)
  pCollection[:] = cp.matmul(pCCAtoCCBMap, pCollection)
  crabCavityKick(pCollection, pLambdaXB, pLambdaYB, kc, phiC)
  pCollection[:] = cp.matmul(L, cp.matmul(pCCBtoIPMap, pCollection))

# Does a cycle around the storage ring with two crab cavities without boosting
#  - Used while ramping up the crab cavity strength
# [input] pCollection: The particle array to track
# [input] pIPtoCCAMap: The transport map from the IP to Crab Cavity A
# [input] pCCAtoCCBMap: The transport map from Crab Cavity A to Crab Cavity B
# [input] pCCBtoIPMap: The transport map from Crab Cavity B to the IP
# [input] pLambdaXA: The strength of Crab Cavity A in the X direction
# [input] pLambdaYA: The strength of Crab Cavity A in the Y direction
# [input] pLambdaXB: The strength of Crab Cavity B in the X direction
# [input] pLambdaYB: The strength of Crab Cavity B in the Y direction
def twoCrabTrackNoBoost(pCollection, pIPtoCCAMap, pCCAtoCCBMap, pCCBtoIPMap, pLambdaXA, pLambdaYA, pLambdaXB,
                        pLambdaYB):
  pCollection[:] = cp.matmul(pIPtoCCAMap, pCollection)
  crabCavityKick(pCollection, pLambdaXA, pLambdaYA, kc, phiC)
  pCollection[:] = cp.matmul(pCCAtoCCBMap, pCollection)
  crabCavityKick(pCollection, pLambdaXB, pLambdaYB, kc, phiC)
  pCollection[:] = cp.matmul(pCCBtoIPMap, pCollection)

# Initialization of transport maps
IPtoCCAMap = transportMap(IPBeta, crabCavityBeta, IPAlpha, crabCavityAlpha, IPtoCCAMu)
CCAtoCCBMap = transportMap(crabCavityBeta, crabCavityBeta, crabCavityAlpha, crabCavityAlpha, CCAtoCCBMu)
CCBtoIPMap = transportMap(crabCavityBeta, IPBeta, crabCavityAlpha, IPAlpha, CCBtoIPMu)

for j in range(int(rampingTurnsStart / rampingInterval), int(rampingTurnsGoal / rampingInterval)):
  # Create a new particle collection whenever a new simulation starts
  collection = get(IPBeta, IPAlpha, epsilon, totalParticles)
  rampingTurns = (j + 1) * rampingInterval
  for i in range(trackingTurns+rampingTurns):
    # First, simulates the ramping up of the crab cavities linearly
    if i < rampingTurns:
      lambdaXA = lambdaXAGoal * ((i + 1.)/rampingTurns)
      lambdaYA = lambdaYAGoal * ((i + 1.)/rampingTurns)
      lambdaXB = lambdaXBGoal * ((i + 1.) / rampingTurns)
      lambdaYB = lambdaYBGoal * ((i + 1.) / rampingTurns)
      twoCrabTrackNoBoost(collection, IPtoCCAMap, CCAtoCCBMap, CCBtoIPMap, lambdaXA, lambdaYA, lambdaXB, lambdaYB)
      if i + 1 == rampingTurns:
        collection[:] = cp.matmul(L, collection)
    # Now, does the normal tracking and also keeps track of emittance for every turn/cycle    
    else:
      lambdaXA = lambdaXAGoal
      lambdaYA = lambdaYAGoal
      lambdaXB = lambdaXBGoal
      lambdaYB = lambdaYBGoal
      twoCrabTrack(collection, IPtoCCAMap, CCAtoCCBMap, CCBtoIPMap, lambdaXA, lambdaYA, lambdaXB, lambdaYB)
      emittanceX[i - rampingTurns] = math.sqrt(cp.linalg.det(cp.cov(collection[0, :], collection[1, :])))
      emittanceY[i - rampingTurns] = math.sqrt(cp.linalg.det(cp.cov(collection[2, :], collection[3, :])))
      emittanceZ[i - rampingTurns] = math.sqrt(cp.linalg.det(cp.cov(collection[4, :], collection[5, :])))
    # Saves the results into a .npz file
    # This will create multiple files depending on how large the ramping range is and how small the ramping interval is
    f = open("CrabCavitySimEmittance" + str(rampingTurns) + "Turns.npz", "wb")
    np.savez(f, emittanceX=emittanceX, emittanceY=emittanceY, emittanceZ=emittanceZ)
    f.close()



