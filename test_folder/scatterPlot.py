import surfaceProject.FeatureVector.featureVector as fv
import surfaceProject.FeatureVector.plotFeature as pf
import surfaceProject.energycalculations.calcenergy as ce

f = []
for i in range(0, 1):
    grid = ce.randSurface(5)
    f.append(grid)

test = fv.getFeatureVectors(f)
pf.plotFeatureMap(test)


