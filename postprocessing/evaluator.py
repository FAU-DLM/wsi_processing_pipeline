import typing
from typing import List, Callable, Tuple, Dict, Union

import postprocessing
import sklearn
from sklearn.metrics import roc_curve, auc, roc_auc_score
import shared

class Evaluator:
    predictor:postprocessing.predictor.Predictor = None
    def __init__(self, predictor:postprocessing.predictor.Predictor):
        self.predictor = predictor
        
    
    def __get_objects_according_to_evaluation_level(self, 
                                                    level:shared.enums.EvaluationLevel, 
                                                    dataset_type:shared.enums.DatasetType) \
                                                    ->List[Union[shared.tile.Tile, \
                                                                 shared.wsi.WholeSlideImage, \
                                                                 shared.case.Case]]:
        objs = None
        if(level == shared.enums.EvaluationLevel.tile):
            objs = self.predictor.patient_manager.get_tiles(dataset_type = dataset_type)
        elif(level == shared.enums.EvaluationLevel.slide):
            objs = self.predictor.patient_manager.get_wsis(dataset_type=dataset_type)
        elif(level == shared.enums.EvaluationLevel.case):
            objs = self.predictor.patient_manager.get_cases(dataset_type=dataset_type)
        else:
            raise ValueError('Wrong value for level.')
            
        return objs
    
    def calculate_accuracy_per_class(self, 
                                     dataset_type:shared.enums.DatasetType, 
                                     level:shared.enums.EvaluationLevel)->Dict[str,float]:
        """
        Calculates how often the model's prediction was correct for each class individually.
        True positive and true negative predictions are seen as correctly predicted and taken into account.
        
        Arguments:
            dataset_type: 
            level: 
            
        Returns:
            Dictionary with the classes as keys and the corresponding accuracies as values
        """
        # tiles, slides or cases according to the specified EvaluationLevel
        objs = self.__get_objects_according_to_evaluation_level(level=level, dataset_type=dataset_type)
         
        #key:class; value: number of correct predictions
        n_correctly_predicted = {}
        for Class in self.predictor.get_classes():
            n_correctly_predicted[Class] = 0
            
        for o in tqdm(objs):
            y_true = o.get_labels()
            y_pred = o.predictions_thresh
            for Class, boolean_value in y_pred.items():
                # if the class appears in the labels and the class was predicted 
                # or the class does not appear in the labels and was not predicted 
                # => correct prediction
                if((Class in y_true and boolean_value) or (Class not in y_true and (not boolean_value))):
                    n_correctly_predicted[Class] += 1
        
        accuracies = {}
        for Class, n in n_correctly_predicted.items():
            # accuracy = number of correctly predicted objects divided by totoal number of objects
            accuracies[Class] = n_correctly_predicted[Class]/len(objs)
            
        return accuracies
    
    def plot_roc_curves(self, 
                        level:shared.enums.EvaluationLevel, 
                        dataset_type:shared.enums.DatasetType):
        """
        Plots roc curves for each class in self.predictor.get_classes().
        """
        objs = self.__get_objects_according_to_evaluation_level(level=level, dataset_type=dataset_type)
        classes = self.predictor.get_classes()
        for Class in tqdm(classes):
            y_preds_raw = [] # list of the predicted percentages
            y_true = [] # list of True and False
            for obj in objs:
                y_preds_raw.append(obj.predictions_raw[Class])
                y_true.append((Class in obj.get_labels()))
                
            
            fpr, tpr, threshold = roc_curve(y_true, y_preds_raw, pos_label=1)
        
            roc_auc = auc(fpr, tpr)
            
            plt.title(f'{Class}')
            plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
            plt.legend(loc = 'lower right')
            plt.plot([0, 1], [0, 1],'r--')
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.ylabel('True Positive Rate')
            plt.xlabel('False Positive Rate')
            plt.show()
            
    def plot_probability_histograms(self, 
                                    level:shared.enums.EvaluationLevel, 
                                    dataset_type:shared.enums.DatasetType):
        """
        Plots probability histograms for each class in self.predictor.get_classes().
        """
        objs = self.__get_objects_according_to_evaluation_level(level=level, dataset_type=dataset_type)
        classes = self.predictor.get_classes()
        
        for Class in tqdm(classes):
            # predicted probabilities of true positives
            probs_true_positive = []
            # predicted probabilities of true negatives
            probs_true_negative = []
            
            for obj in objs:
                predicted_prob = obj.predictions_raw[Class]
                if(Class in obj.get_labels()):
                    probs_true_positive.append(predicted_prob)
                else:
                    probs_true_negative.append(predicted_prob)
            
            from matplotlib import pyplot
            
            bins = np.linspace(0, 1, 50)
            
            pyplot.hist(probs_true_positive, bins, alpha=0.5, label='true positive')
            pyplot.hist(probs_true_negative, bins, alpha=0.5, label='true negative')
            pyplot.legend(loc='upper right')
            pyplot.title(label=Class)
            pyplot.ylabel('Frequency')
            pyplot.xlabel('Predicted Probability')
            pyplot.show()