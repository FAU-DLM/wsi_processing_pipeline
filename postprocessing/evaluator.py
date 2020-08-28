import typing
from typing import List, Callable, Tuple, Dict, Union

import postprocessing
import shared

from tqdm import tqdm
import sklearn
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
import numpy
import numpy as np
import fastai
import fastcore

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
    
    
    def __get_y_true_and_y_pred(self, 
                                level:shared.enums.EvaluationLevel, 
                                dataset_type:shared.enums.DatasetType):
        
        objs = self.__get_objects_according_to_evaluation_level(level=level, dataset_type=dataset_type)
        
        y_true = []
        y_pred = []
        for obj in objs:
            y_true.append(obj.get_labels_one_hot_encoded())
            y_pred.append(obj.get_predictions_one_hot_encoded())
        
        return y_true, y_pred
    
    
    def confusion_matrix(self, 
                        level:shared.enums.EvaluationLevel, 
                        dataset_type:shared.enums.DatasetType):
        y_true, y_pred = self.__get_y_true_and_y_pred(level=level, dataset_type=dataset_type)
        cms = sklearn.metrics.multilabel_confusion_matrix(y_true=y_true, 
                                                         y_pred=y_pred)
            
        return cms
    
    
    def plot_confusion_matrix(self, 
                              level:shared.enums.EvaluationLevel, 
                              dataset_type:shared.enums.DatasetType):
        cms = self.confusion_matrix(level=level, dataset_type=dataset_type)
        vocab = self.predictor.get_classes()
        for n, cm in enumerate(cms):
            fig = plt.figure()
            plt.imshow(cm, interpolation='nearest', cmap='Blues')
            plt.title(vocab[n])
            tick_marks = np.arange(2)
            plt.xticks(tick_marks, ['positive','negative'], rotation=90)
            plt.yticks(tick_marks, ['positive','negative'], rotation=0)
            plt.tight_layout()
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            plt.grid(False)
            
            for row in range(cm.shape[0]):
                for col in range(cm.shape[1]):
                    thresh = cm.max() / 2.
                    plt.text(x = col, 
                             y = row, 
                             s=cm[row][col], 
                             horizontalalignment="center", 
                             verticalalignment="center", 
                             color="white" if cm[row, col] > thresh else "black")
    
    def classification_report(self, 
                              level:shared.enums.EvaluationLevel, 
                              dataset_type:shared.enums.DatasetType):
        y_true, y_pred = self.__get_y_true_and_y_pred(level=level, dataset_type=dataset_type)
        print(sklearn.metrics.classification_report(y_true,y_pred))
        
    def grad_cam(self, 
                 image:fastai.vision.core.PILImage,
                 thresholds:List[float] = None,
                 take_predicted_class_indices:bool = True,
                 class_indices:List[int] = None, 
                 model_layer = None):
        """
        Plots image with grad_cam overlay.
        Arguments:
            image: e.g. learner.dls.valid_ds[0][0]
            thresholds: One threshold for every class to determine predictions from the raw output percentages.
            take_predicted_class_index: The grad_cam heatmap can be calculated for every possible class. If this value
                                        is true, it will be calculated for those classes whose prediction scores passed
                                        the threshold. 
            class_indices: Only relevant if <take_predicted_class_indices> is set to False. You can specify for which classes
                            the grad-cam heatmaps shall be calculated. See self.predictor.get_classes() or 
                            learner.dls.vocab for the class order.
            model_layer: The grad_cam heatmaps can be calculated for every layer of the model's body. By default this 
                         library takes the last convolutional layer of the model's body, if no layer is specified.
                         example of specifieng a layer: learner.model[0][-1] == last layer of model's body
                         
        Returns:
            Plots the specfied image with grad-cam heatmap overlay.
        """
        if((not take_predicted_class_indices) and (class_indices is None or len(class_indices) == 0)):
            raise ValueError('You have to specify class indices if take_predicted_class_indices is set to False.')
        if((not take_predicted_class_indices) and class_indices is not None):
            if(min(class_indices) < 0 or max(class_indices) >= len(self.predictor.get_classes())):
                raise ValueError('Values of class indices must be in range [0, len(self.predictor.get_classes()))')
        if(thresholds is None):
            thresholds = numpy.repeat(0.5, len(self.predictor.get_classes()))
        if (model_layer == None):
            model_layer = self.predictor.learner.model[0][-1]
        
        
        class Hook():
            def __init__(self, m):
                self.hook = m.register_forward_hook(self.hook_func)   
            def hook_func(self, m, i, o): self.stored = o.detach().clone()
            def __enter__(self, *args): return self
            def __exit__(self, *args): self.hook.remove()
        
        class HookBwd():
            def __init__(self, m):
                self.hook = m.register_backward_hook(self.hook_func)   
            def hook_func(self, m, gi, go): self.stored = go[0].detach().clone()
            def __enter__(self, *args): return self
            def __exit__(self, *args): self.hook.remove()
                
        x, = fastcore.utils.first(self.predictor.learner.dls.test_dl([image]))
        x_dec = fastai.torch_core.TensorImage(self.predictor.learner.dls.train.decode((x,))[0][0])

        with HookBwd(self.predictor.learner.model[0]) as hookg:
            with Hook(self.predictor.learner.model[0]) as hook:
                output = self.predictor.learner.model.eval()(x.cuda())
                act = hook.stored
                
                classes = []
                if(take_predicted_class_indices):
                    preds = torch.sigmoid(output).cpu()[0].detach().numpy() >= thresholds
                    for n, class_name in enumerate(self.predictor.get_classes()):
                        if(preds[n]):
                            classes.append(class_name)
                
                elif(not take_predicted_class_indices):
                    for i in class_indices:
                        classes.append(self.predictor.get_classes()[i])
                
                else: assert False
                
                for n, class_name in enumerate(classes):
                    output[0,n].backward(retain_graph=True)
                    grad = hookg.stored
                    w = grad[0].mean(dim=[1,2], keepdim=True)
                    cam_map = (w * act[0]).sum(0)
                    _,ax = plt.subplots()
                    ax.title.set_text(class_name)
                    x_dec.show(ctx=ax)
                    ax.imshow(cam_map.detach().cpu(), alpha=0.6, extent=(0,512,512,0),
                              interpolation='bilinear', cmap='magma');