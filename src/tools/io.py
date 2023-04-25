import tensorflow as tf

def data_resize(x,y,shape):
    px = []
    py = []

    for i,j in zip(x,y):
        px.append(tf.expand_dims(tf.image.resize(i,size=shape,method='nearest'),axis=0))
        py.append(tf.expand_dims(tf.image.resize(j,size=shape,method='nearest'),axis=0))

    return tf.concat(px,axis=0),tf.concat(py,axis=0)