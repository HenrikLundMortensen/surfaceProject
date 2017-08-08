import surfaceProject.FeatureVector.findStructureWithFeature as fs
import surfaceProject.FeatureVector.calcEnergyWithFeature as cewf

set,energy = fs.generateTraining(5,3)
print('The energy of the three first grids are:',cewf.EBondFeatureGrid(set[0]),cewf.EBondFeatureGrid(set[1]),cewf.EBondFeatureGrid(set[2]))
print('The energies are:', energy)
print('The surfaces are:',set)
