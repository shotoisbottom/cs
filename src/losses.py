import tensorflow as tf

def SSIM(x,y):
    return tf.image.ssim(x,y,1.0)

def PSNR(x,y):
    return tf.image.psnr(x,y,1.0)

def L1(x,y):
    return tf.keras.losses.mae(x,y)

def L2(x,y):
    return tf.keras.losses.mse(x,y)

def MS_SSIM(x,y):
    return tf.image.ssim_multiscale(x,y,1.0)

def loss_funtion(pred_im , gt , name="L1"):
    if name == "L1":
        return L1(pred_im, gt)

    elif name == "L2":
        return L2(pred_im, gt)

    elif name == "SSIM":
        return 1-SSIM(pred_im, gt)
