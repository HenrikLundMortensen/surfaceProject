import matplotlib.pyplot as plt
import surfaceProject.energycalculations.findStructure as fs
import surfaceProject.FeatureVector.findStructureWithFeature as fswf
import surfaceProject.plotGrid.plotGrid as pg

# Plot animation
#fswf.findOptimumAnimation(5)
#plt.show()

# Plot correct structure
surface = fs.findOptimum(5)
fig = pg.initializePlotGridFigure(5)
pg.plotGrid(surface, fig)
plt.show()
