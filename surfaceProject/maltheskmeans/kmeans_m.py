import numpy as np
import surfaceProject.FeatureVector.featureVector as fv
import surfaceProject.energycalculations.calcenergy as ce

N=5
grid = ce.randSurface(N)
f = fv.getBondFeatureVectorsSingleGrid(grid)
print(f)


