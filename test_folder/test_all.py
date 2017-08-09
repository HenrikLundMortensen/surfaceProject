import surfaceProject.energycalculations.findStructure as fs
import surfaceProject.energycalculations.calcenergy as ce


class TestSimpleGridImplementation():
    def test_SurfaceFind(self):
        # Test that we find the correct structure
        surface = fs.findOptimum(5)
        assert ce.calculateEnergy(surface, 5) == - 48

    def test_sizeOfGrid(self):
        surface = ce.randSurface(7)
        assert surface.size == 7*7
