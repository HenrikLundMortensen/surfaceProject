import surfaceProject.FeatureVector.featureVector as fv
import surfaceProject.FeatureVector.plotFeature as pf
import surfaceProject.energycalculations.findStructure as fs
import surfaceProject.energycalculations.calcenergy as ce
import numpy as np

f = []
for i in range(0,2):
    grid = ce.randSurface(5)
    f.append(grid)

test = fv.getFeatureVectors(f)
pf.plotFeatureMap(test)


