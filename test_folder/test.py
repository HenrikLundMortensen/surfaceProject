import surfaceProject.energycalculations.calcenergy as ce
import surfaceProject.featureVector.featureVectors as fv
surf = ce.randSurface(5)
print(surf)
#f = fv.getBondFeatureVectorsSingleGrid(surf)
#print(f)

f = fv.getBondFeatureVectorsSingleGrid(surf)
print(f)
