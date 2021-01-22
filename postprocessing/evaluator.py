import typing
from typing import List, Callable, Tuple, Dict, Union

import postprocessing
import shared
import visualization
from visualization.gradcam import GradCam
from visualization.guided_gradcam import GuidedGradCam

from tqdm import tqdm
import sklearn
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
import numpy
import numpy as np
import torch
import fastai
import fastcore
import math
import sklearn
import pandas
import pandas as pd
import PIL
import cv2

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
            #print(o.path)
            y_true = o.get_labels()
            #print(f'y_true: {y_true}')
            y_pred = o.predictions_thresh
            #print(f'y_pred: {y_pred}')
            y_pred_raw = o.predictions_raw
            #print(f'y_pred_raw: {y_pred_raw}')            
            for Class, boolean_value in y_pred.items():
                # if the class appears in the labels and the class was predicted 
                # or the class does not appear in the labels and was not predicted 
                # => correct prediction
                if((Class in y_true and boolean_value) or (Class not in y_true and (not boolean_value))):
                    #print('INSIDE IF')
                    #print(f'class: {Class}')
                    #print(f'bool: {boolean_value}')
                    #print('')
                    n_correctly_predicted[Class] += 1
                    
                    
            #print('')        
            #print('------------------------------------------------------------------------------------')
            #print('')
               
        #print(f'n_correctly_predicted: {n_correctly_predicted}')
        
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
            
    def get_tiles_with_top_losses(self, 
                                  dataset_type:shared.enums.DatasetType, 
                                  k:int = 10, 
                                  descending:bool = True)->List[shared.tile.Tile]:
        """
        Returns the k tiles from the specified dataset with the highest or if descending == False with the lowest loss.
        """
        tls = self.predictor.patient_manager.get_tiles(dataset_type = dataset_type)
        tls.sort(key=lambda tile: tile.loss, reverse=descending)
        return tls[:k]
    
    
    def __plot_figures(self, figures, ncols, figsize=(18,18)):
        # https://stackoverflow.com/questions/11159436/multiple-figures-in-a-single-window
        """Plot a dictionary of figures.
    
        Parameters
        ----------
        figures : <title, figure> dictionary
        ncols : number of columns of subplots wanted in the display
        nrows : number of rows of subplots wanted in the figure
        """
        nrows = math.ceil(len(figures)/ncols)
        fig, axeslist = plt.subplots(ncols=ncols, nrows=nrows, figsize=figsize)

        for i in range(0, nrows*ncols):
            if(i < len(figures)):
                axeslist.ravel()[i].imshow(list(figures.values())[i], cmap=plt.gray())
                axeslist.ravel()[i].set_title(list(figures.keys())[i])
            axeslist.ravel()[i].set_axis_off()
        plt.subplots_adjust(left = None, bottom=None, right=None, top=None, wspace=None, hspace=None)
    
    def plot_top_losses(self, dataset_type:shared.enums.DatasetType, k:int = 10, descending:bool = True):
        """
        Plots the k tiles from the specified dataset with the highest or if descending == False with the lowest loss.
        """
        tls = self.get_tiles_with_top_losses(dataset_type=dataset_type, k=k, descending=descending)
        
        ###
        # dataframe
        ###
        df = pd.DataFrame(columns=['patient', 'slide', 'target', 'predicted', 'probabilities', 'loss'])
        for t in tls:
            predicted = []
            for Class, bool_value in t.predictions_thresh.items():
                if(bool_value):
                    predicted.append(Class)
            df = df.append({'patient':t.roi.whole_slide_image.case.patient.patient_id, 
                       'slide':t.roi.whole_slide_image.slide_id, 
                       'target':t.labels, 
                       'predicted':predicted, 
                       'probabilities':t.predictions_raw, 
                       'loss':t.loss}, ignore_index=True)
        fastai.torch_core.display_df(df)
        # Alternative 
        #https://stackoverflow.com/questions/26873127/show-dataframe-as-table-in-ipython-notebook
        #from IPython.display import display, HTML
        
        ###
        # images
        ###
        tile_images = [t.get_pil_image() for t in self.predictor.patient_manager.get_all_tiles()[:k]]
        figures = {}
        for n, img in enumerate(tile_images):
            figures[n] = img
        self.__plot_figures(figures=figures, ncols=4)
    
    def __calculate_metric(self, 
                            obj:Union[shared.case.Case, 
                            shared.wsi.WholeSlideImage, 
                            shared.tile.Tile], 
                            metric:Callable)->float:
        """
        Arguments:
            obj: shared.case.Case or shared.wsi.WholeSlideImage or shared.tile.Tile 
            metric: Callable that takes y_pred and y_true as arguments
        """
        y_pred = list(obj.predictions_raw.values())
        y_true = obj.get_labels_one_hot_encoded()
        return metric(y_pred=y_pred, y_true=y_true)
        
    
    def get_orderd_by_metric(self, 
                          dataset_type:shared.enums.DatasetType, 
                          level:shared.enums.EvaluationLevel, 
                          k:int = 10, 
                          metric:Callable=sklearn.metrics.mean_absolute_error, 
                          descending:bool = True)->List[Union[shared.case.Case, 
                                                                shared.wsi.WholeSlideImage, 
                                                                shared.tile.Tile]]:
        """
        This method compares the real labels with the predicted probabilities and returns the k cases/slides/tiles 
        (depending on level) where the predicted probabilities differ the most/least 
        (depending on "descending") from the real labels using metric.
        Example for standard value metric = sklearn.metrics.mean_absolute_error:
            two classes: A and B
            a case/slide/tile with the target [0,1] and the predicted probabilities of [0.2, 0.8]
            Difference between reality and prediction is now calculated the following way:
            (abs(0-0.2)+abs(1-0.8))/2
            
        Arguments:
            dataset_type:
            level:
            k: number of objects to return
            metric: metric to order the objects
            descending:
        Returns:
            The k objects with the highest metric values
        """
        objs = self.__get_objects_according_to_evaluation_level(dataset_type=dataset_type, level=level)
        for obj in objs:
            obj.metric = self.__calculate_metric(obj=obj, metric=metric)
        objs.sort(key=lambda o: o.metric, reverse=descending)
        return objs[:k]
    
    def get_df_of_k_top_ordered_by_metric(self, 
                                          dataset_type:shared.enums.DatasetType, 
                                          level:shared.enums.EvaluationLevel, 
                                          k:int = 10, 
                                          metric:Callable=sklearn.metrics.mean_absolute_error, 
                                          descending:bool = True)->pandas.DataFrame:
        """
        This method compares the real labels with the predicted probabilities and returns the k cases/slides/tiles 
        (depending on level) where the predicted probabilities differ the most/least 
        (depending on "descending") from the real labels using metric.
        Example for standard value metric = sklearn.metrics.mean_absolute_error:
            two classes: A and B
            a case/slide/tile with the target [0,1] and the predicted probabilities of [0.2, 0.8]
            Difference between reality and prediction is now calculated the following way:
            (abs(0-0.2)+abs(1-0.8))/2
            
        Arguments:
            dataset_type:
            level:
            k: number of objects to return
            metric: metric to order the objects
            descending:
        Returns:
            a dataframe with information about the k objects with the highest metric values
        """
        objs = self.get_orderd_by_metric(dataset_type=dataset_type, 
                                         level=level, 
                                         k=k, 
                                         metric=metric, 
                                         descending=descending)
        ###
        # dataframe
        ###
        level_name = str(level).split('.')[1]
        df = pd.DataFrame(columns=[level_name, 'target', 'predicted', 'probabilities', 'metric'])
        for o in objs:
            predicted = []
            for Class, bool_value in o.predictions_thresh.items():
                if(bool_value):
                    predicted.append(Class)
            df = df.append({level_name:o,  
                            'target':o.get_labels(), 
                            'predicted':predicted, 
                           'probabilities':o.predictions_raw, 
                           'metric':o.metric}, 
                           ignore_index=True)
        return df

    
    ###
    #  previous functioning version. just keeping it here for a while, if there might be a bug in the new version.
    ###
    #def grad_cam(self, 
    #             tile:shared.tile.Tile,
    #             grad_cam_result:shared.enums.GradCamResult = shared.enums.GradCamResult.predicted,
    #             thresholds:List[float] = None,                 
    #             class_indices:List[int] = None, 
    #             model_layer = None):
    #    """
    #    Plots image with grad_cam overlay.
    #    Arguments:
    #        tile: one of patient_manager's tiles 
    #        thresholds: One threshold for every class to determine predictions from the raw output percentages.
    #        grad_cam_result: The grad_cam heatmap can be calculated for every possible class.
    #                         shared.enums.GradCamResult.predicted: grad-cams for the classes, whose prediction score were above
    #                                                                 the given threshold
    #                         shared.enums.GradCamResult.targets: if targets are available (tile.get_labels()), grad-cams for the 
    #                                                             classes that are part of the target classes are shown
    #        class_indices: Only relevant if <grad_cam_result> is set to None. You can specify for which classes
    #                        the grad-cam heatmaps shall be calculated. See self.predictor.get_classes() or 
    #                        learner.dls.vocab for the class order.
    #        model_layer: The grad_cam heatmaps can be calculated for every layer of the model's body. By default this 
    #                     library takes the last convolutional layer of the model's body, if no layer is specified.
    #                     example of specifieng a layer: learner.model[0][-1] == last layer of model's body
    #                     
    #    Returns:
    #        Plots the specfied image with grad-cam heatmap overlay.
    #    """
    #    if(grad_cam_result is None and (class_indices is None or len(class_indices) == 0)):
    #        raise ValueError('You have to specify class indices if grad_cam_result is set to None.')
    #    if(grad_cam_result is None and class_indices is not None):
    #        if(min(class_indices) < 0 or max(class_indices) >= len(self.predictor.get_classes())):
    #            raise ValueError('Values of class indices must be in range [0, len(self.predictor.get_classes()))')
    #    if(thresholds is None):
    #        thresholds = numpy.repeat(0.5, len(self.predictor.get_classes()))
    #    if(model_layer == None):
    #        model_layer = self.predictor.learner.model[0][-1]
    #        
    #    if(grad_cam_result == shared.enums.GradCamResult.targets and (tile.get_labels() == None or len(tile.get_labels())==0)):
    #       raise ValueError
    #    
    #    
    #    class Hook():
    #        def __init__(self, m):
    #            self.hook = m.register_forward_hook(self.hook_func)   
    #        def hook_func(self, m, i, o): self.stored = o.detach().clone()
    #        def __enter__(self, *args): return self
    #        def __exit__(self, *args): self.hook.remove()
    #    
    #    class HookBwd():
    #        def __init__(self, m):
    #            self.hook = m.register_backward_hook(self.hook_func)   
    #        def hook_func(self, m, gi, go): self.stored = go[0].detach().clone()
    #        def __enter__(self, *args): return self
    #        def __exit__(self, *args): self.hook.remove()
    #            
    #    x, = fastcore.utils.first(self.predictor.learner.dls.test_dl([tile]))
    #    x_dec = fastai.torch_core.TensorImage(self.predictor.learner.dls.train.decode((x,))[0][0])
#
    #    with HookBwd(model_layer) as hookg:
    #        with Hook(model_layer) as hook:
    #            output = self.predictor.learner.model.eval()(x.cuda())
    #            act = hook.stored
    #            
    #            predicted_classes = []
    #            preds_raw = torch.sigmoid(output).cpu()[0].detach().numpy()
    #            preds =  preds_raw >= thresholds
    #            for n, class_name in enumerate(self.predictor.get_classes()):
    #                    if(preds[n]):
    #                        predicted_classes.append(class_name)
    #       
    #            classes_to_show = []
    #            if(grad_cam_result == shared.enums.GradCamResult.predicted):
    #                classes_to_show = predicted_classes
    #                
    #            elif(grad_cam_result == shared.enums.GradCamResult.targets):
    #                classes_to_show = tile.get_labels()
    #            elif(grad_cam_result == None):
    #                for i in class_indices:
    #                    classes_to_show.append(self.predictor.get_classes()[i])
    #            
    #            else: assert False
    #            
    #            preds_dict = {}
    #            for class_name, pred in zip(self.predictor.learner.dls.vocab, preds_raw):
    #                preds_dict[class_name] = pred.item()
    #            print(f'predicted percentages {preds_dict}')
    #            print(f'predicted classes: {predicted_classes}')
    #            if(tile.get_labels() != None and len(tile.get_labels()) > 0):
    #                print(f'targets: {tile.get_labels()}')
    #            for n, class_name in enumerate(classes_to_show):
    #                output[0,n].backward(retain_graph=True)
    #                grad = hookg.stored
    #                w = grad[0].mean(dim=[1,2], keepdim=True)
    #                cam_map = (w * act[0]).sum(0)
    #                _,ax = plt.subplots()
    #                ax.title.set_text(class_name)
    #                x_dec.show(ctx=ax)
    #                ax.imshow(cam_map.detach().cpu(), alpha=0.6, extent=(0,512,512,0),
    #                          interpolation='bilinear', cmap='magma');
    
    
    
    def __common_trunk(self, 
                         tile:shared.tile.Tile,
                         grad_cam_result:shared.enums.GradCamResult = shared.enums.GradCamResult.predicted,
                         thresholds:List[float] = None,                 
                         class_indices:List[int] = None, 
                         model_layer = None):
        
        if(grad_cam_result is None and (class_indices is None or len(class_indices) == 0)):
            raise ValueError('You have to specify class indices if grad_cam_result is set to None.')
        if(grad_cam_result is None and class_indices is not None):
            if(min(class_indices) < 0 or max(class_indices) >= len(self.predictor.get_classes())):
                raise ValueError(f'Values of class indices must be in range [0, {len(self.predictor.get_classes())})')
        if(thresholds is None):
            thresholds = numpy.repeat(0.5, len(self.predictor.get_classes()))
        if(model_layer == None):
            model_layer = self.predictor.learner.model[0][-1]           
        if(grad_cam_result == shared.enums.GradCamResult.targets \
           and (tile.get_labels() == None or len(tile.get_labels())==0)):
            raise ValueError('no labels available for the specified tile')
            
        self.predictor.learner.model.cpu()
                
        x, = fastcore.utils.first(self.predictor.learner.dls.test_dl([tile]))
        x_dec = fastai.torch_core.TensorImage(self.predictor.learner.dls.train.decode((x,))[0][0])

        output = self.predictor.learner.model.eval()(x.cpu())
                
        predicted_classes = []
        preds_raw = torch.sigmoid(output).cpu()[0].detach().numpy()
        preds =  preds_raw >= thresholds
        for n, class_name in enumerate(self.predictor.get_classes()):
            if(preds[n]):
                predicted_classes.append(class_name)
           
        classes_to_show = []
        if(grad_cam_result == shared.enums.GradCamResult.predicted):
            classes_to_show = predicted_classes                    
        elif(grad_cam_result == shared.enums.GradCamResult.targets):
            classes_to_show = tile.get_labels()
        elif(grad_cam_result == None):
            labels = self.predictor.get_classes()
            for i in class_indices:
                classes_to_show.append(labels[i])               
        else: assert False
                
        preds_dict = {}
        for class_name, pred in zip(self.predictor.learner.dls.vocab, preds_raw):
            preds_dict[class_name] = pred.item()
        print(f'predicted percentages {preds_dict}')
        print(f'predicted classes: {predicted_classes}')
        if(tile.get_labels() != None and len(tile.get_labels()) > 0):
            print(f'targets: {tile.get_labels()}')
            
        return x, x_dec, classes_to_show
    
    def grad_cams_separately(self, 
                 tile:shared.tile.Tile,
                 grad_cam_result:shared.enums.GradCamResult = shared.enums.GradCamResult.predicted,
                 thresholds:List[float] = None,                 
                 class_indices:List[int] = None, 
                 model_layer = None)->Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Returns the tile image with a grad_cam for every class depending on the specified grad_cam_result parameter.
        Arguments:
            tile: one of patient_manager's tiles 
            grad_cam_result: The grad_cam heatmap can be calculated for every possible class.
                             shared.enums.GradCamResult.predicted: grad-cams for the classes, whose prediction score were above
                                                                     the given threshold
                             shared.enums.GradCamResult.targets: if targets are available (tile.get_labels()), grad-cams for the 
                                                                 classes that are part of the target classes are shown
            
            thresholds: One threshold for every class to determine predictions from the raw output percentages.
            
            class_indices: Only relevant if <grad_cam_result> is set to None. You can specify for which classes
                            the grad-cam heatmaps shall be calculated. See self.predictor.get_classes() or 
                            learner.dls.vocab for the class order.
            model_layer: The grad_cam heatmaps can be calculated for every layer of the model's body. By default this 
                         library takes the last convolutional layer of the model's body, if no layer is specified.
                         example of specifieng a layer: learner.model[0][-1] == last layer of model's body
                         
        Returns:           
            Tuple:
                First position: The denormalized tile rgb image as a tensor with shape [channels, height, width]
                Second position: Dictionary with the class names as keys and the grad_cam maps (torch.Tensor with shape 
                                    [height, width]) as values
        """
        x, x_dec, classes_to_show = self.__common_trunk(tile=tile, 
                                                        grad_cam_result=grad_cam_result, 
                                                        thresholds=thresholds, 
                                                        class_indices=class_indices, 
                                                        model_layer=model_layer)        
        cam_maps = {}
        
        grad_cam_extractor = GradCam(model = self.predictor.learner.model, model_layer = model_layer)
        vocab = list(self.predictor.get_classes())
        for class_name in classes_to_show:
            cam_map = grad_cam_extractor.generate_cam(input_image = x, class_index = vocab.index(class_name))
            cam_maps[class_name] = cam_map
            
        return x_dec, cam_maps
    
    def apply_grad_cam(self, img:numpy.ndarray, grad_cam_map:numpy.ndarray)->numpy.ndarray:
        """
        Arguments:
            img: denormed RGB image with shape [channels, height, width] as numpy array
            grad_cam_map: 2-dimensional numpy array with the same height and width as the img
        Returns:
            img as numpy array with the grad_cam_map as semitransparent overlay 
            in shape [height, width, channels] in RGB format
        """
        def denorm(arr):
            return ((arr - arr.min()) * (1/(arr.max() - arr.min()) * 255)).astype('uint8')
        
        img = img.transpose(1,2,0)
        cv_img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR)
        grad_cam = denorm(grad_cam_map).astype(np.uint8)
        heatmap = cv2.applyColorMap(grad_cam, cv2.COLORMAP_MAGMA)
        img_with_heatmap = cv2.addWeighted(heatmap, 0.6, cv_img, 0.4, 0)
        return cv2.cvtColor(img_with_heatmap, cv2.COLOR_BGR2RGB)
    
    
    def grad_cams_merged(self, 
                 tile:shared.tile.Tile,
                 grad_cam_result:shared.enums.GradCamResult = shared.enums.GradCamResult.predicted,
                 thresholds:List[float] = None,                 
                 class_indices:List[int] = None, 
                 model_layer = None)->Dict[str, numpy.ndarray]:
        """
        Returns the tile image merged with a grad_cam for every class depending on the specified grad_cam_result parameter.
        Arguments:
            tile: one of patient_manager's tiles 
            grad_cam_result: The grad_cam heatmap can be calculated for every possible class.
                             shared.enums.GradCamResult.predicted: grad-cams for the classes, whose prediction score were above
                                                                     the given threshold
                             shared.enums.GradCamResult.targets: if targets are available (tile.get_labels()), grad-cams for the 
                                                                 classes that are part of the target classes are shown
            
            thresholds: One threshold for every class to determine predictions from the raw output percentages.
            
            class_indices: Only relevant if <grad_cam_result> is set to None. You can specify for which classes
                            the grad-cam heatmaps shall be calculated. See self.predictor.get_classes() or 
                            learner.dls.vocab for the class order.
            model_layer: The grad_cam heatmaps can be calculated for every layer of the model's body. By default this 
                         library takes the last convolutional layer of the model's body, if no layer is specified.
                         example of specifieng a layer: learner.model[0][-1] == last layer of model's body
                         
        Returns:           
            Dictionary with class_names as keys and images as numpy array with the grad_cam_map as semitransparent overlay 
            in shape [height, width, channels] in RGB format
        """
        class_name_to_img_with_grad_cam_overlay = {}
        x_dec, cam_maps = self.grad_cams_separately(tile=tile, 
                                                    grad_cam_result=grad_cam_result, 
                                                    thresholds=thresholds, 
                                                    class_indices=class_indices, 
                                                    model_layer=model_layer)
        x_dec_numpy = x_dec.numpy()
        for class_name, cam_map in cam_maps.items():
            class_name_to_img_with_grad_cam_overlay[class_name] = self.apply_grad_cam(img=x_dec_numpy,
                                                                                      grad_cam_map=cam_map.numpy())
        return class_name_to_img_with_grad_cam_overlay
 

    def plot_grad_cams(self, 
                 tile:shared.tile.Tile,
                 grad_cam_result:shared.enums.GradCamResult = shared.enums.GradCamResult.predicted,
                 thresholds:List[float] = None,                 
                 class_indices:List[int] = None, 
                 model_layer = None, 
                 figsize=(5,5)):
        """
        Plots image with grad_cam overlay.
        Arguments:
            tile: one of patient_manager's tiles 
            thresholds: One threshold for every class to determine predictions from the raw output percentages.
            grad_cam_result: The grad_cam heatmap can be calculated for every possible class.
                             shared.enums.GradCamResult.predicted: grad-cams for the classes, whose prediction score were above
                                                                     the given threshold
                             shared.enums.GradCamResult.targets: if targets are available (tile.get_labels()), grad-cams for the 
                                                                 classes that are part of the target classes are shown
            class_indices: Only relevant if <grad_cam_result> is set to None. You can specify for which classes
                            the grad-cam heatmaps shall be calculated. See self.predictor.get_classes() or 
                            learner.dls.vocab for the class order.
            model_layer: The grad_cam heatmaps can be calculated for every layer of the model's body. By default this 
                         library takes the last convolutional layer of the model's body, if no layer is specified.
                         example of specifieng a layer: learner.model[0][-1] == last layer of model's body  
            figsize: size of the resulting plot
            
        Returns:
            Plots the specfied image with grad-cam heatmap overlay for every class 
            depending on the specified grad_cam_result.
        """                        
        x_dec, grad_cam_maps = self.grad_cams_separately(tile=tile, 
                                         grad_cam_result=grad_cam_result, 
                                         thresholds=thresholds, 
                                         class_indices=class_indices, 
                                         model_layer=model_layer)
        
        for class_name, cam_map in grad_cam_maps.items():
            _,ax = plt.subplots(figsize=figsize)
            ax.title.set_text(class_name)
            x_dec.show(ctx=ax)
            ax.imshow(cam_map.detach().cpu(), alpha=0.6, interpolation='bilinear', cmap='magma');
            
    def guided_grad_cams(self, 
                         tile:shared.tile.Tile,
                         grad_cam_result:shared.enums.GradCamResult = shared.enums.GradCamResult.predicted,
                         thresholds:List[float] = None,                 
                         class_indices:List[int] = None, 
                         model_layer = None)->Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Returns the tile image with a guided grad_cam for every class depending on the specified grad_cam_result parameter.
        Arguments:
            tile: one of patient_manager's tiles 
            thresholds: One threshold for every class to determine predictions from the raw output percentages.
            grad_cam_result: The guided grad-cam mask can be calculated for every possible class.
                             shared.enums.GradCamResult.predicted: grad-cams for the classes, whose prediction score 
                                                                     were above the given threshold
                             shared.enums.GradCamResult.targets: if targets are available (tile.get_labels()), 
                                                                 grad-cams for the 
                                                                 classes that are part of the target classes are shown
            class_indices: Only relevant if <grad_cam_result> is set to None. You can specify for which classes
                            the grad-cam heatmaps shall be calculated. See self.predictor.get_classes() or 
                            learner.dls.vocab for the class order.
            model_layer: To calculate the guided grad-cam mask, a grad-cam is calculated. 
                         The grad_cam can be calculated for every layer of the model's body. By default this 
                         library takes the last convolutional layer of the model's body, if no layer is specified.
                         example of specifieng a layer: learner.model[0][-1] == last layer of model's body
            figsize: size of the resulting plot
            
        Returns:
            Tuple:
                First position: The denormalized tile rgb image as a tensor with shape [channels, height, width]
                Second position: Dictionary with the class names as keys and the guided 
                                    grad_cam maps (torch.Tensor with shape [height, width]) as values
        """
        x, x_dec, classes_to_show = self.__common_trunk(tile=tile, 
                                                        grad_cam_result=grad_cam_result, 
                                                        thresholds=thresholds, 
                                                        class_indices=class_indices, 
                                                        model_layer=model_layer)
            
        GGC = GuidedGradCam(self.predictor.learner.model)
        vocab = list(self.predictor.get_classes())
        
        guided_grad_cam_maps = {}
        for class_name in classes_to_show:
            guided_grad_cam_mask = GGC.generate_guided_grad_cam(input_image=x, 
                                                                class_index=vocab.index(class_name), 
                                                                normalize=True)           
            guided_grad_cam_maps[class_name] = guided_grad_cam_mask
            
        return x_dec, guided_grad_cam_maps
            
    def plot_guided_grad_cams(self, 
                         tile:shared.tile.Tile,
                         grad_cam_result:shared.enums.GradCamResult = shared.enums.GradCamResult.predicted,
                         thresholds:List[float] = None,                 
                         class_indices:List[int] = None, 
                         model_layer = None, 
                         figsize = (10,10)):
        """
        Plots image of guided grad-cam for an image and a certain class.
        Arguments:
            tile: one of patient_manager's tiles 
            thresholds: One threshold for every class to determine predictions from the raw output percentages.
            grad_cam_result: The guided grad-cam mask can be calculated for every possible class.
                             shared.enums.GradCamResult.predicted: grad-cams for the classes, whose prediction score 
                                                                     were above the given threshold
                             shared.enums.GradCamResult.targets: if targets are available (tile.get_labels()), 
                                                                 grad-cams for the 
                                                                 classes that are part of the target classes are shown
            class_indices: Only relevant if <grad_cam_result> is set to None. You can specify for which classes
                            the grad-cam heatmaps shall be calculated. See self.predictor.get_classes() or 
                            learner.dls.vocab for the class order.
            model_layer: To calculate the guided grad-cam mask, a grad-cam is calculated. 
                         The grad_cam can be calculated for every layer of the model's body. By default this 
                         library takes the last convolutional layer of the model's body, if no layer is specified.
                         example of specifieng a layer: learner.model[0][-1] == last layer of model's body
            figsize: size of the resulting plot
            
        Returns:
            Plots the original HE tile with guided grad cam maps for the different classes
        """
        x_dec, guided_grad_cam_maps = self.guided_grad_cams(tile=tile, 
                                         grad_cam_result=grad_cam_result, 
                                         thresholds=thresholds, 
                                         class_indices=class_indices, 
                                         model_layer=model_layer)
        
        _, ax = plt.subplots(figsize=figsize)
        ax.title.set_text('HE')
        ax.imshow(tile.get_pil_image())
        
        for class_name, guided_grad_cam in guided_grad_cam_maps.items():         
            _,ax = plt.subplots(figsize=figsize)
            ax.title.set_text(class_name)
            ax.imshow(guided_grad_cam.permute(1,2,0))