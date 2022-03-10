import tensorflow.keras.backend as K


def ccc_loss(gold, pred):  # Concordance correlation coefficient (CCC)-based loss function - using non-inductive statistics
    # input (num_batches, seq_len, 1)
    gold       = K.squeeze(gold, axis=-1)
    pred       = K.squeeze(pred, axis=-1)
    gold_mean  = K.mean(gold, axis=-1, keepdims=True)
    pred_mean  = K.mean(pred, axis=-1, keepdims=True)
    covariance = (gold-gold_mean)*(pred-pred_mean)
    gold_var   = K.mean(K.square(gold-gold_mean), axis=-1, keepdims=True)
    pred_var   = K.mean(K.square(pred-pred_mean), axis=-1, keepdims=True)
    ccc        = K.constant(2.) * covariance / (gold_var + pred_var + K.square(gold_mean - pred_mean) + K.epsilon())
    ccc_loss   = K.constant(1.) - ccc
    return ccc_loss

def ccc_loss_multiple_targets(gold, pred):  # Concordance correlation coefficient (CCC)-based loss function - using non-inductive statistics
    # input (num_batches, seq_len, 1)
    #gold       = K.squeeze(gold, axis=-1)
    #pred       = K.squeeze(pred, axis=-1)
    gold_mean  = K.mean(gold, axis=1, keepdims=True)
    #gold_mean = K.mean(gold_mean, axis=1, keepdims=True)
    pred_mean  = K.mean(pred, axis=1, keepdims=True)
    #pred_mean = K.mean(pred_mean, axis=1, keepdims=True)
    covariance = (gold-gold_mean)*(pred-pred_mean)
    gold_var   = K.mean(K.square(gold-gold_mean), axis=1, keepdims=True)
    pred_var   = K.mean(K.square(pred-pred_mean), axis=1, keepdims=True)
    ccc        = K.constant(2.) * covariance / (gold_var + pred_var + K.square(gold_mean - pred_mean) + K.epsilon())
    ccc_mean = K.mean(ccc)
    ccc_loss   = K.constant(1.) - ccc_mean
    return ccc_loss

def ccc(gold, pred):
    loss = ccc_loss(gold, pred)
    return K.constant(1.) - loss