import surfaceProject.energycalculations.calcenergy as ce
import surfaceProject.FeatureVector.featureVector as fv
surf = ce.randSurface(5)
print(surf)
f = fv.getBondFeatureVectorsSingleGrid(surf)
print(f)
