from abc import ABCMeta, abstractmethod

# abstract base class for defining labels
class Label:
    __metaclass__ = ABCMeta

    @abstractmethod
    def __str__(self): pass

       
class ClassificationLabel(Label):
    def __init__(self, label):
        self._label = label

    def __str__(self):
        # TODO
        pass

class FeatureVector:
    def __init__(self):
        self._feature_vector = []
        
    def add(self, index, value):
            # Fill in missing features with 0
            for i in range(len(self._feature_vector), index):
                self._feature_vector.append(0)

            self._feature_vector.append(value)
        
    def get(self, index):
        return self._feature_vector[index]
        

class Instance:
    def __init__(self, feature_vector, label):
        self.feature_vector = feature_vector
        self.label = label

# abstract base class for defining predictors
class Predictor:
    __metaclass__ = ABCMeta

    @abstractmethod
    def train(self, instances): pass

    @abstractmethod
    def predict(self, instance): pass

       
