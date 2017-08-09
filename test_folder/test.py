import surfaceProject.energycalculations.findStructure as fs
import surfaceProject.FeatureVector.featureVector as fv

surf = fs.findOptimum(5)
f = fv.getBondFeatureVectorsSingleGrid(surf)
print(f)
