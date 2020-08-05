import tensorflow as tf
import keras
keras.backend.set_image_data_format('channels_first')
from keras.layers import Conv2D, Add, Lambda,Activation,add
from keras.engine.topology import Layer

class pixelshuffle(Layer):
    """Sub-pixel convolution layer.
    See https://arxiv.org/abs/1609.05158
    """
    def __init__(self, scale, trainable=False, **kwargs):
        self.scale = scale
        super().__init__(trainable=trainable, **kwargs)

    def call(self, t):
        
        upscale_factor = self.scale
        input_size = t.shape.as_list()
        dimensionality = len(input_size) - 2
        new_shape = self.compute_output_shape(input_size)
        C = new_shape[1]
        
        output_size = new_shape[2:]        
        x = [upscale_factor] * dimensionality
        old_h = input_size[-2] if input_size[-2] is not None else -1
        old_w = input_size[-1] if input_size[-1] is not None else -1
        
        shape = tf.shape( t )
        t = tf.reshape(t,[-1, C, x[0], x[1], shape[-2],shape[-1]])
        
        perms = [0, 1, 5, 2, 4, 3]
        t = tf.transpose(t, perm=perms) 
        t = Lambda( self.squeeze_middle2axes_operator, output_shape = self.squeeze_middle2axes_shape , arguments={'C':C,'output_size':output_size}) (t)
        t = tf.transpose(t, [0, 1, 3, 2] )
        return t

    def squeeze_middle2axes_operator( self, x4d , C,output_size) :
        shape = tf.shape( x4d ) # get dynamic tensor shape
        x4d = tf.reshape( x4d, [shape[0],shape[1] ,shape[2]*2,shape[4]*2 ] )
        return x4d

    def squeeze_middle2axes_shape( self, output_size  ) :
        in_batch ,C, in_rows,_,in_cols,_ = output_size
       
        if ( None in [ in_rows, in_cols] ) :
            output_shape = ( in_batch, C,  None, None )
        else :
            output_shape = ( in_batch, C, in_rows, in_cols  )
        return output_shape

    def compute_output_shape(self, input_shape):
        r = self.scale
        rrC ,H, W= np.array(input_shape[1:])
        assert rrC % (r ** 2) == 0
        height = H * r if H is not None else -1
        width = W * r if W is not None else -1
        
        return (input_shape[0], rrC // (r ** 2), height, width)

   

