import torch
from torch.nn import ReLU
import fastcore

class GuidedBackprop():
    """
       Produces gradients generated with guided back propagation from the given image
    """
    def __init__(self, model):
        #store hooks that need to be removed at the end
        self.hooks = []
       
        self.model = model
        self.gradients = None
        self.forward_relu_outputs = []

    def __update_relus(self):
        """
            Updates relu activation functions so that
                1- stores output in forward pass
                2- imputes zero for gradient values that are less than zero
        """
        def __relu_backward_hook_function(module, grad_in, grad_out):
            """
            If there is a negative gradient, change it to zero
            """
            # Get last forward output
            corresponding_forward_output = self.forward_relu_outputs[-1]
            corresponding_forward_output[corresponding_forward_output > 0] = 1
            modified_grad_out = corresponding_forward_output * torch.clamp(grad_in[0], min=0.0)
            del self.forward_relu_outputs[-1]  # Remove last forward output
            return (modified_grad_out,)

        def __relu_forward_hook_function(module, ten_in, ten_out):
            """
            Store results of forward pass
            """
            self.forward_relu_outputs.append(ten_out)

        # Loop through layers, hook up ReLUs
        for module in self.model[0].modules():
            if isinstance(module, ReLU):
                self.hooks.append(module.register_backward_hook(__relu_backward_hook_function))
                self.hooks.append(module.register_forward_hook(__relu_forward_hook_function))
        
    def __hook_first_layer(self):
        def __hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]
        # Register hook to the first layer
        first_layer = self.model[0][0]
        self.hooks.append(first_layer.register_backward_hook(__hook_function))
       
    def generate_gradients(self, input_image:torch.Tensor, class_index:int)->torch.Tensor:
        """
        Arguments:
            input_image: normalized image with shape [batch_size, channels, height, width]
            class_index: e.g. you got four classes [A,B,C,D] class_index can be one of the numbers
                         0, 1, 2, 3
        Return:
            gradient mask as torch.Tensor in shape [channels, height, width] of original input image
        """
        self.model.cpu()
        self.model.eval()
        self.model.requires_grad_()
        input_image = input_image.cpu()
        input_image.requires_grad_()
        
        try:
            self.__update_relus()
            self.__hook_first_layer()
            
            # Forward pass
            model_output = self.model(input_image)
            # Zero gradients
            self.model.zero_grad()
            # Target for backprop
            one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
            one_hot_output[0][class_index] = 1        
            # Backward pass
            model_output.backward(gradient=one_hot_output)
            # Convert Pytorch variable to numpy array
            # [0] to get rid of the first channel (1,3,224,224)
            gradients_as_arr = self.gradients.data.numpy()[0]
            
        finally:
            #remove hooks
            for h in self.hooks:
                h.remove()
            self.hooks.clear()
              
        return torch.from_numpy(gradients_as_arr)