import spatialNN
from spatialNN import Model
from PIL import Image

model = Model.Spatial_Model(
        n_dimensions = 2, 
        n_neurons = 50, 
        inputs = 3, 
        outputs = 2,
    )

im = Image.fromarray(model.draw())
im.save("test_model.png")