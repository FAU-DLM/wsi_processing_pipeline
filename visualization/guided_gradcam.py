from visualization.gradcam import GradCam
from visualization.guided_backprop import GuidedBackprop
import torch

class GuidedGradCam:
    def __init__(self, 
                 model:torch.nn.modules.container.Sequential, 
                 model_layer:torch.nn.modules.container.Sequential = None):
        self.model = model
        if(model_layer == None):
            model_layer = model[0][-1]
        self.model_layer = model_layer
        
    def generate_guided_grad_cam(self, input_image:torch.Tensor, class_index:int, denorm:bool)->torch.Tensor:
        """
        Arguments:
            input_image: normalized image with shape [batch_size, channels, height, width]
            class_index: e.g. you got four classes [A,B,C,D] class_index can be one of the numbers
                         0, 1, 2, 3
            denorm: if True, the mask gets denormed
        Return:
            guided grad-cam as torch.Tensor in shape [channels, height, width] of original input image
        """
        self.model.cpu()
        input_image = input_image.cpu()
        
        GC = GradCam(self.model)
        grad_cam_mask = GC.generate_cam(input_image=input_image, class_index=class_index)
        
        GBP = GuidedBackprop(self.model)
        guided_backprop_mask = GBP.generate_gradients(input_image=input_image, class_index=class_index)
        
        guided_grad_cam_mask = torch.mul(grad_cam_mask, guided_backprop_mask)
        if(denorm):
            guided_grad_cam_mask = (guided_grad_cam_mask - guided_grad_cam_mask.min())/guided_grad_cam_mask.max()
            
        return guided_grad_cam_mask