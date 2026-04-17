#%%
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

#%%
def resnet(input_shape = (64, 64, 3), input_tensor = None, weights = "None"):
    base = ResNet50(weights = weights, include_top = False, input_tensor = input_tensor, input_shape = input_shape)
    x = GlobalAveragePooling2D()(base.output)
    output = Dense(10, activation = "softmax", name = "before_softmax")(x)
    model = Model(inputs = base.input, outputs = output)

    return model