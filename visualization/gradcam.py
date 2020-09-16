import torch
import torch.nn.functional as F

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


class GradCam():
    """
        Produces class activation map
    """
    def __init__(self, 
                 model:torch.nn.modules.container.Sequential, 
                 model_layer:torch.nn.modules.container.Sequential = None):
        """
        Arguments:
            model: a pytorch model
            target_layer: one layer of the model; e.g. fastai.learner.model[0][-1]
        """
        
        self.model = model
        if(model_layer == None):
            model_layer = model[0][-1]
        self.model_layer = model_layer

    def generate_cam(self, input_image:torch.Tensor, class_index:int)->torch.Tensor:
        """
        Arguments:
            input_image: normalized image with shape [batch_size, channels, height, width]
            class_index: e.g. you got four classes [A,B,C,D] class_index can be one of the numbers
                         0, 1, 2, 3
        Return:
            class activation map as torch.Tensor in shape [channels, height, width] of original input image
        """
        self.model.cpu()
        input_image = input_image.cpu()
        with HookBwd(self.model_layer) as hookg:
            with Hook(self.model_layer) as hook:
                output = self.model.eval()(input_image)
                act = hook.stored
                output[0,class_index].backward(retain_graph=True)
                grad = hookg.stored
                w = grad[0].mean(dim=[1,2], keepdim=True)
                cam_map = (w * act[0]).sum(0)
                with torch.no_grad():
                    cam_map.unsqueeze_(0)
                    cam_map.unsqueeze_(0)
                cam_map = F.interpolate(cam_map, size=input_image.shape[-1], mode='bilinear', align_corners=True)
                return cam_map.squeeze()