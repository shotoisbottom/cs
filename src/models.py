import tensorflow as tf

class CBAM(tf.keras.layers.Layer):
    def __init__(self, trainable=True , filter_num = [3,3,3] ,reduction_ratio = 8 , stride = 1, name = "CBAM" , *args, **kwargs,):
        super(CBAM,self).__init__(trainable, name, *args, **kwargs)
        self.conv1 = tf.keras.layers.Conv2D(filter_num[0], (1,1), strides=stride, padding='same', name=name+'_conv1')
        self.bn1 = tf.keras.layers.BatchNormalization(name=name+"_bn1")
        self.relu1 = tf.keras.layers.Activation('relu', name=name+'_relu1')

        self.conv2 = tf.keras.layers.Conv2D(filter_num[1], (3,3), strides=1, padding='same', name=name+'_conv2')
        self.bn2 = tf.keras.layers.BatchNormalization(name=name+"_bn2")
        self.relu2 = tf.keras.layers.Activation('relu', name=name+'_relu2')

        self.conv3 = tf.keras.layers.Conv2D(filter_num[2], (1,1), strides=1, padding='same', name=name+'_conv3')
        self.bn3 = tf.keras.layers.BatchNormalization(name=name+"_bn3")

        self.channel_avgpool = tf.keras.layers.GlobalAveragePooling2D(name=name+'_channel_avgpool')
        self.channel_maxpool = tf.keras.layers.GlobalMaxPool2D(name=name+'_channel_maxpool')

        self.channel_fc1 = tf.keras.layers.Dense(filter_num[2]//reduction_ratio, activation='relu', name=name+'_channel_fc1')
        self.channel_fc2 = tf.keras.layers.Dense(filter_num[2], activation='relu', name=name+'_channel_fc2')

        self.channel_sigmod = tf.keras.layers.Activation('sigmoid', name=name+'_channel_sigmoid')
        self.channel_reshape = tf.keras.layers.Reshape((1,1,filter_num[2]), name=name+'_channel_reshape')

        self.spatial_conv2d = tf.keras.layers.Conv2D(1, (7,7), strides=1, padding='same',name=name+'_spatial_conv2d')

        self.spatial_sigmoid = tf.keras.layers.Activation('sigmoid', name=name+'_spatial_sigmoid')

        self.residual = tf.keras.layers.Conv2D(filter_num[2], (1,1), strides=stride, padding='same', name=name+'_residual')

        self.relu3 = tf.keras.layers.Activation('relu', name=name+'_relu3')

    def call(self, inputs, training=None, mask=None):
        x = inputs
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.bn3(x)

        # Channel Attention
        avgpool = self.channel_avgpool(x)
        maxpool = self.channel_maxpool(x)
        # Shared MLP
        avg_out = self.channel_fc2(self.channel_fc1(avgpool))
        max_out = self.channel_fc2(self.channel_fc1(maxpool))

        channel = tf.keras.layers.add([avg_out, max_out])
        channel = self.channel_sigmod(channel)
        channel = self.channel_reshape(channel)
        channel_out = tf.multiply(x, channel)
    
        # Spatial Attention
        avgpool = tf.reduce_mean(channel_out, axis=3, keepdims=True)
        maxpool = tf.reduce_max(channel_out, axis=3, keepdims=True)
        spatial = tf.keras.layers.Concatenate(axis=3)([avgpool, maxpool])

        spatial = self.spatial_conv2d(spatial)
        spatial_out = self.spatial_sigmoid(spatial)

        CBAM_out = tf.multiply(channel_out, spatial_out)

        # residual connection
        r = self.residual(inputs)
        x = tf.keras.layers.add([CBAM_out, r])
        x = self.relu3(x)

        return x

class Conv(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, dilation_rate=1, strides=1, padding='same', act = 'silu', bn = True, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super(Conv, self).__init__(trainable, name, dtype, dynamic, **kwargs)
        self.conv = tf.keras.Sequential()
        self.conv.add(tf.keras.layers.Conv2D(
                                                filters=filters,
                                                kernel_size=kernel_size, 
                                                dilation_rate=dilation_rate, 
                                                strides=strides, 
                                                padding=padding,
                                                use_bias=not(bn)
            ))
        if bn != False:
            self.conv.add(tf.keras.layers.BatchNormalization())
        if act != None:
            self.conv.add(tf.keras.layers.Activation(act))

    def call(self, inputs, *args, **kwargs):
        return self.conv(inputs)
    

class BasicModule(tf.keras.layers.Layer):
    def __init__(self, len, filters, kernel_size, act = 'relu',trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super(BasicModule, self).__init__(trainable, name, dtype, dynamic, **kwargs)
        self.conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=1)
        self.basic = tf.keras.Sequential([Conv(filters=filters, kernel_size=kernel_size, act=act) for _ in range(len)])

    def call(self, inputs, *args, **kwargs):
        inputs = self.conv(inputs)
        return inputs + self.basic(inputs)


class UnetProMax(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super(UnetProMax, self).__init__(*args, **kwargs)
        filters = [16,64,256]
        self.conv = [Conv(filters=3, kernel_size=1, bn=False, act='relu') for _ in range(2)]
        self.Down = tf.keras.layers.MaxPool2D(pool_size=2, strides=2)
        self.Up = tf.keras.layers.UpSampling2D(size=2, interpolation='nearest')

        self.Dbasic = [Conv(kernel_size=3, filters=num) for num in filters]
        self.Ubasic = [Conv(kernel_size=3, filters=num) for num in filters[::-1]]

        self.Dcbam = [CBAM(filter_num=[num,num,num]) for num in filters]
        self.Ucbam = [CBAM(filter_num=[num,num,num]) for num in filters[::-1]]
    
    def call(self, inputs, training=None, mask=None):
        x = self.conv[0](inputs)

        l1 = self.Dbasic[0](x)
        l1 = self.Dcbam[0](l1)

        l2 = self.Dbasic[1](self.Down(l1))
        l2 = self.Dcbam[1](l2)

        l3 = self.Dbasic[2](self.Down(l2))
        l3 = self.Dcbam[2](l3)

        r3 = self.Ucbam[0](l3)
        r3 = self.Ubasic[0](r3)

        r2 = self.Ucbam[0](l3)
        r2 = self.Ubasic[1](self.Up(r2))

        r1 = self.Ucbam[0](l2)
        r1 = self.Ubasic[2](self.Up(r1))

        x = self.conv[-1](r1)
        return x + inputs


if __name__ == "__main__":
    model = UnetProMax()

    model.compile(loss=tf.keras.losses.mse,
                  optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  metrics=tf.keras.losses.mse)
    
    model.build(input_shape=[4,256,256,3])