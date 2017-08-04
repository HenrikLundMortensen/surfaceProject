import numpy as np
import FeatureVector.featureVector as fv
import calcenergy as ce

N=5
grid = ce.randSurface(N)
f = fv.getBondFeatureVectorsSingleGrid(grid)
print(f)


