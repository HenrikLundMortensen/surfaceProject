import surfaceProject.FeatureVector.findStructureWithFeature as fs

training,test = fs.generateTraining(5,10,0.3)
print('The training set is:',training)
print('The test set is:',test)
