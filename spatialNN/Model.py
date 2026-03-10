import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import Module
from torch.nn import Parameter
from torch import Tensor
import typing

class Spatial_Model(Module):
    ''' 
    Neural network model in which neurons are placed in 
    some spatial volume and weights depend on the distances between them.
    Input and output nodes are represented by fixed points in this space.
    
    '''

    def __init__(
            self,
            n_dimensions: int,
            n_neurons: int,
            inputs: int,
            outputs: int,
            activation: typing.Callable[[Tensor], Tensor] = torch.nn.Hardsigmoid(),
            input_positions: Tensor = None,
            output_positions: Tensor = None,
            distance_weight_scaling: float = 1.0,
            distance_threshold: float = 0.2,
            distance_thresh_fn: typing.Callable[[Tensor], Tensor] = torch.nn.Hardsigmoid(),
            input_init_fn: typing.Callable[[typing.List[int]], Tensor] = torch.rand,
            output_init_fn: typing.Callable[[typing.List[int]], Tensor] = torch.rand,
            init_fn: typing.Callable[[typing.List[int]], Tensor] = torch.rand,
            recurrent_mode: bool = False
        ) -> None:
        
        super().__init__()
        
        self._activation = activation
        self._n_neurons: int = n_neurons
        self._n_dimensions: int = n_dimensions
        self._distance_thresh_fn: typing.Callable[[Tensor], Tensor] = distance_thresh_fn
        self._distance_thresh_fn.__init__()
        self._distance_weight_scaling: float = distance_weight_scaling
        self._distance_threshold: float = distance_threshold
        self._recurrent_mode = recurrent_mode

        ## create a buffer to store the state of the hidden neuron
        self.register_buffer("hidden_state", torch.zeros([n_neurons]))

        ## initialise the hidden neuron positions based the initialiser function
        self.register_parameter(name = "positions", param = Parameter(init_fn([n_neurons, n_dimensions])))
        
        ## If user specified input and output positions, first check that they're valid then set 
        if input_positions != None:
            self._input_positions: Tensor = self._validate_positions(input_positions, inputs, n_dimensions)
        
        ## If they didn't specify then let's set them
        else:
            self._input_positions: Tensor = input_init_fn(inputs, n_dimensions)

        ## same again for output positions
        if output_positions != None:
            self._output_positions = self._validate_positions(output_positions, outputs, n_dimensions)
            
        else:
            self._output_positions: Tensor = output_init_fn(outputs, n_dimensions)

    def forward(self, input: Tensor) -> Tensor:

        if self._recurrent_mode:

            retTensors = []

            if input.ndim == 1:
                raise ValueError("Too few dimensions for recurrent mode, should be 2 (time steps, features), or three (batch, time steps, features)")
            
            elif input.ndim == 2:
                input_list = torch.tensor_split(input, input.shape[0], dim=0)
                batch_size = 0
                stack_dim = 0
                
            elif input.ndim == 3:
                batch_size = input.shape[0]
                if batch_size != 1:
                    raise NotImplementedError("Sorry I can only do one dimension at the moment :(")
                input_list = torch.tensor_split(input, input.shape[1], dim=1)
                stack_dim = 0

            for inp in input_list:
                retTensors.append(self._single_forward(torch.squeeze(inp)))

            ret = torch.stack(retTensors, stack_dim)
            if batch_size > 0:
                ret = torch.unsqueeze(ret, 0)

            return ret

        else:
            ret = self._single_forward(input)
            return ret
        
    def _single_forward(self, input: Tensor) -> Tensor:

        input_weights = self._get_weights(self.get_parameter("positions"), self._input_positions)
        hidden_weights = self._get_weights(self.get_parameter("positions"), self.get_parameter("positions"))
        output_weights = self._get_weights(self._output_positions, self.get_parameter("positions"))

        new_hidden = self.get_buffer("hidden_state") + torch.matmul(hidden_weights, self.get_buffer("hidden_state")) + torch.matmul(input_weights, input)
        #print(new_hidden.shape)
        self.hidden_state[...] = self._activation(new_hidden)
        
        """
        print("input weights:", input_weights, "\nshape =", input_weights.shape)
        print("hidden weights:", hidden_weights, "\nshape =", hidden_weights.shape)
        print("output weights:", output_weights, "\nshape =", output_weights.shape)

        print("BLAAAA")
        print(self.get_buffer("hidden_state"))
        print(torch.matmul(hidden_weights, self.get_buffer("hidden_state")))
        
        print("new hidden:", new_hidden)
        print("new hidden shape:", new_hidden.shape)
        """

        return torch.matmul(output_weights, self.get_buffer("hidden_state"))
    
    def _get_weights(self, output: Tensor, input: Tensor) -> Tensor:
        '''
        Calculate the weights based on the current positions of the neurons
        '''

        distances = torch.cdist(output, input)

        ## the 3.0 scaling is because the sigmoid functions extend out to +- 3 range which is beyond the normal
        ## bounds of our space so we squeeze it in a bit
        return self._distance_weight_scaling * self._distance_thresh_fn( self._distance_threshold - 3.0 * distances)
    
    def _validate_positions(to_validate: Tensor, n_vals_expected: int, n_dims_expected: int) -> Tensor:
        ''' 
        Validate that a specified tensor has the number of values and dimensions that are expected.
        If it's all good then the tensor is returned
        '''

        ## first check that the first dim matches the number of input values expected
        assert to_validate.shape().numpy()[0] == n_vals_expected 
        ## then check that the number of dimensions matches the number specified
        assert to_validate.shape().numpy().shape[0] == n_dims_expected

        ## if alls well then return 
        return to_validate
    
    def _draw_connections(self, plt_axis, positions1: Tensor, positions2: Tensor) -> None:
        ''' 
        Draw the connections between two sets of neurons
        '''
        
        weights = self._get_weights(positions1, positions2).detach().numpy()

        n1 = positions1.detach().numpy().shape[0]
        n2 = positions2.detach().numpy().shape[0]

        for i in range(n1):
            for j in range(n2):
                plt_axis.plot(
                    [positions1.numpy()[i, 0], positions2.numpy()[j, 0]],
                    [positions1.numpy()[i, 1], positions2.numpy()[j, 1]],
                    c = "k", 
                    lw = max(0, weights[i,j] / self._distance_weight_scaling - self._distance_threshold)
                )


    def _draw_pyplot(self, plt_axis) -> None:
        '''
        render the layer to a pyplot axis object
        '''

        assert self._n_dimensions == 2

        hidden_positions = self.get_parameter("positions").detach()
        input_positions = self._input_positions #get_parameter("input_positions").detach()
        output_positions = self._output_positions #get_parameter("output_positions").detach()

        self._draw_connections(plt_axis, hidden_positions, hidden_positions)
        self._draw_connections(plt_axis, input_positions, hidden_positions)
        self._draw_connections(plt_axis, hidden_positions, output_positions)

        hidden = self.get_buffer("hidden_state").detach().numpy()

        plt_axis.scatter(hidden_positions.numpy()[..., 0], hidden_positions.numpy()[..., 1], c = hidden)
        plt_axis.scatter(input_positions.numpy()[..., 0], input_positions.numpy()[..., 1], c = "b", label = "inputs")
        plt_axis.scatter(output_positions.numpy()[..., 0], output_positions.numpy()[..., 1], c = "r", label = "outputs")
    
        plt.legend()
    
    def draw(self, plt_axis=None, figsize=None):
        '''
        render the layer to a numpy array of RGB values 
        '''

        if(plt_axis != None):
            self._draw_pyplot(plt_axis)
        
        else:
            fig = plt.figure(0, dpi=160, figsize=figsize)
            ax_0 = fig.add_subplot(111)

            self._draw_pyplot(ax_0)
            fig.canvas.draw()
            
            # Now we can save it to a numpy array.
            data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

            fig.clf()

            return data