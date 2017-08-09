import surfaceProject.energycalculations.findStructure as fs
import surfaceProject.plotGrid.plotGrid as pg
import matplotlib.pyplot as plt


surface = fs.findOptimum(5)
fig = pg.initializePlotGridFigure(5)
pg.plotGrid(surface, fig)
plt.show()

