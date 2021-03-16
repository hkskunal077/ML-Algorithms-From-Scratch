def titanicKNN():
  import matplotlib.pyplot as plt
  import random, numpy

  def minkowskiDist(v1, v2, p):
      dist = 0.0
      for i in range(len(v1)):
          dist += abs(v1[i]-v2[i])**p
      return dist**(1/p)


  class Animal(object):
      def __init__(self, name, features):
          """Assumes name a string; features a list of numbers"""
          self.name = name
          self.features = numpy.array(features)

      def getName(self):
          return self.name

      def getFeatures(self):
          return self.features

      def distance(self, other):
          """Assumes other an Animal
             Returns the Euclidean distance between feature vectors
                of self and other"""
          return minkowskiDist(self.getFeatures(), other.getFeatures(), 2)

      def __str__(self):
          return self.name

  def actualNumLegsAnimals():
      #Actual number of legs
      cobra = Animal('cobra', [1,1,1,1,0])
      rattlesnake = Animal('rattlesnake', [1,1,1,1,0])
      boa = Animal('boa\nconstrictor', [0,1,0,1,0])
      chicken = Animal('chicken', [1,1,0,1,2])
      alligator = Animal('alligator', [1,1,0,1,4])
      dartFrog = Animal('dart frog', [1,0,1,0,4])
      zebra = Animal('zebra', [0,0,0,0,4])
      python = Animal('python', [1,1,0,1,0])
      guppy = Animal('guppy', [0,1,0,0,0])
      animals = [cobra, rattlesnake, boa, chicken, guppy, dartFrog, zebra, python, alligator]
      return animals

  def binaryFeatureAnimals():
      ##Binary features only           
      cobra = Animal('cobra', [1,1,1,1,0])
      rattlesnake = Animal('rattlesnake', [1,1,1,1,0])
      boa = Animal('boa\nconstrictor', [0,1,0,1,0])
      chicken = Animal('chicken', [1,1,0,1,2])
      alligator = Animal('alligator', [1,1,0,1,1])
      dartFrog = Animal('dart frog', [1,0,1,0,1])
      zebra = Animal('zebra', [0,0,0,0,1])
      python = Animal('python', [1,1,0,1,0])
      guppy = Animal('guppy', [0,1,0,0,0])
      animals = [cobra, rattlesnake, boa, chicken, guppy, dartFrog, zebra, python, alligator]
      return animals

  #Instead of Nearest Neighbour we take k Nearest Neighbour,
  #because one bad neighbour will influence, We choose corresponding k
  #so we assume the k Nearest is a better option
  #To make sure that we have a winnder, we take an odd K for disbalance

  def compareAnimals(animals, precision):
      """Assumes animals is a list of animals, precision an int >= 0
         Builds a table of Euclidean distance between each animal"""
      #Get labels for columns and rows
      columnLabels = []
      for a in animals:
          columnLabels.append(a.getName())
      rowLabels = columnLabels[:]
      tableVals = []
      #Get distances between pairs of animals
      #For each row
      for a1 in animals:
          row = []
          #For each column
          for a2 in animals:
              if a1 == a2:
                  row.append('--')
              else:
                  distance = a1.distance(a2)
                  row.append(str(round(distance, precision)))
          tableVals.append(row)
      #Produce table
      table = plt.table(rowLabels = rowLabels,
                          colLabels = columnLabels,
                          cellText = tableVals,
                          cellLoc = 'center',
                          loc = 'center')
      table.auto_set_font_size(False)
      table.set_fontsize(10)
      table.scale(1, 2.5)
      plt.axis('off')
      plt.savefig('distances')

  ##compareAnimals(actualNumLegsAnimals(), 3)
  ##plt.show()
  ##compareAnimals(binaryFeatureAnimals(), 3)
  ##plt.show()
  #assert False
  #Accuracy is not a great measure
  #Lets say a disease is 0.1%
  #Then ML model saying NO you wont have it
  #will have accuracy of 99.99%
  #But the meaning of model fails, BAD ML

  class Passenger(object):
      featureNames = ('C1', 'C2', 'C3', 'age', 'male gender')
      def __init__(self, pClass, age, gender, survived, name):
          self.name = name
          self.featureVec = [0, 0, 0, age, gender]
          self.featureVec[pClass - 1] = 1
          self.label = survived
          self.cabinClass = pClass
      def distance(self, other):
          return minkowskiDist(self.featureVec, other.featureVec, 2)
      def getClass(self):
          return self.cabinClass
      def getAge(self):
          return self.featureVec[3]
      def getGender(self):
          return self.featureVec[4]
      def getName(self):
          return self.name
      def getFeatures(self):
          return self.featureVec[:]
      def getLabel(self):
          return self.label


  def getTitanicData(fname):
      data = {}
      data['class'], data['survived'], data['age'] = [], [], []
      data['gender'], data['name'] = [], []

      f = open(fname)
      line = f.readline()
      while line != '':
          split = line.split(',')
          data['class'].append(int(split[0]))
          data['age'].append(float(split[1]))

          if split[2] == 'M':
              data['gender'].append(1)
          else:
              data['gender'].append(0)

          if split[3] == '1':
              data['survived'].append('Survived')
          else:
              data['survived'].append('Died')

          data['name'].append(split[:])
          line = f.readline()
      return data


  def buildTitanicExamples(fileName):
      data = getTitanicData(fileName)
      examples = []
      for i in range(len(data['class'])):

          p = Passenger(data['class'][i], data['age'][i],
                        data['gender'][i], data['survived'][i],
                        data['name'][i])
          examples.append(p)
          print("Finished Processing ", len(examples), " passengers\n")
          return examples


  examples = buildTitanicExamples('TitanicPassengers.txt')


  def findNearest(name, exampleSet, metric):
      for e in exampleSet:
          if e.getName() == name:
              example = e
              break
      curDist = None
      for e in exampleSet:
          if e.getName() != name:
              if curDist == None or metric(example, e) < curDist:
                  nearest = e
                  curDist = metric(example, nearest)
      return nearest


  def accuracy(truePos, falsePos, trueNeg, falseNeg):
      numerator = truePos + trueNeg
      denominator = truePos + trueNeg + falsePos + falseNeg
      try:
          return numerator/denominator
      except ZeroDivisionError:
          return float('nan')

  def sensitivity(truePos, falseNeg):
      try:
          return truePos/(truePos + falseNeg)
      except ZeroDivisionError:
          return float('nan')

  def specificity(trueNeg, falsePos):
      try:
          return trueNeg/(trueNeg + falsePos)
      except ZeroDivisionError:
          return float('nan')

  def posPredVal(truePos, falsePos):
      try:
          return truePos/(truePos + falsePos)
      except ZeroDivisionError:
          return float('nan')

  def negPredVal(trueNeg, falseNeg):
      try:
          return trueNeg/(trueNeg + falseNeg)
      except ZeroDivisionError:
          return float('nan')


  def getStats(truePos, falsePos, trueNeg, falseNeg, toPrint = True):

      accur = accuracy(truePos, falsePos, trueNeg, falseNeg)
      sens = sensitivity(truePos, falseNeg)
      spec = specificity(trueNeg, falsePos)
      ppv = posPredVal(truePos, falsePos)

      if toPrint:
          print(' Accuracy =', round(accur, 3))
          print(' Sensitivity =', round(sens, 3))
          print(' Specificity =', round(spec, 3))
          print(' Pos. Pred. Val. =', round(ppv, 3))
      return (accur, sens, spec, ppv)



  def findKNearest(example, exampleSet, k):
      kNearest = []
      distances = []
      for i in range(k):
          kNearest.append(exampleSet[i])
          distances.append(example.distance(exampleSet[i]))
      maxDist = max(distances)

      for e in exampleSet[k:]:
          dist = example.distance(e)
          if dist < maxDist:
              maxIndex = distances.index(maxDist)
              kNearest[maxIndex] = e
              distances[maxIndex] = dist
              maxDist = max(distances)

      return kNearest, distances



  def KNearestClassify(training, testSet, label, k):

      truePos, falsePos, trueNeg, falseNeg = 0, 0, 0, 0
      for testCase in testSet:
          nearest, distances = findKNearest(testCase, training, k)
          #Conduct a VOTE
          numMatch = 0
          for i in range(len(nearest)):
              if nearest[i].getLabel() == label:
                  numMatches += 1
          if numMatch > k//2:
              if testCase.getLabel() == label:
                  truePos += 1
              else:
                  falsePos += 1
          else:
              if testCase.getLabel() == label:
                  trueNeg += 1
              else:
                  falseNeg += 1
      return truePos, falsePos, trueNeg, falseNeg



  def leaveOneOut(examples, method, toPrint = True):
      truePos, falsePos, trueNeg, falseNeg = 0, 0, 0, 0
      for i in range(len(examples)):
          testCase = examples[i]
          trainingData = examples[0:i] + examples[i+1:]
          results = method(trainingData, [testCase])
          truePos += results[0]
          falsePos += results[1]
          trueNeg += results[2]
          falseNeg += results[3]
      if toPrint:
          getStats(truePos, falsePos, trueNeg, falseNeg)
      return truePos, falsePos, trueNeg, falseNeg


  def split80_20(examples):
      sampleIndices = random.sample(range(len(examples)),
                                    len(examples)//5)
      trainingSet, testSet = [], []
      for i in range(len(examples)):
          if i in sampleIndices:
              testSet.append(examples[i])
          else:
              trainingSet.append(examples[i])
      return trainingSet, testSet


  def randomSplits(examples, method, numSplits, toPrint = True):
      truePos, falsePos, trueNeg, falseNeg = 0, 0, 0, 0
      random.seed(0)
      for t in range(numSplits):
          trainingSet, testSet = split80_20(examples)
          results = method(trainingSet, testSet)
          truePos += results[0]
          falsePos += results[1]
          trueNeg += results[2]
          falseNeg += results[3]
      getStats(truePos/numSplits, falsePos/numSplits,
               trueNeg/numSplits, falseNeg/numSplits, toPrint)
      return truePos/numSplits, falsePos/numSplits,\
               trueNeg/numSplits, falseNeg/numSplits



  knn = lambda training, testSet:\
                KNearestClassify(training, testSet, 'Survived', 3)

  numSplits = 10
  print('Average of', numSplits,
        '80/20 splits using KNN (k=3)')
  truePos, falsePos, trueNeg, falseNeg =\
        randomSplits(examples, knn, numSplits)

  print('Average of LOO testing using KNN (k=3)')
  truePos, falsePos, trueNeg, falseNeg =\
        leaveOneOut(examples, knn)
