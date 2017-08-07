from setuptools import setup

setup(name='surfaceProject',
      version='1.0',
      description='Package for discrete surface calculations',
      author='Malthe Kjær Bisbo ',
      author_email='mkb@phys.au.dk',
      packages=['surfaceProject'],
      packages=['surfaceProject','surfaceProject.energycalculations','surfaceProject.FeatureVector','surfaceProject.PlotGrid','surfaceProject.maltheskmeans','surfaceProject.henriksKmeans'],
      zip_safe=False)
