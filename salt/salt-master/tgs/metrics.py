import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.losses import binary_crossentropy


def cast(var):
    return K.cast(var, dtype='float32')


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred = K.cast(y_pred, 'float32')
    y_pred_f = K.cast(K.greater(K.flatten(y_pred), 0.5), 'float32')
    intersection = y_true_f * y_pred_f
    score = 2. * K.sum(intersection) / (K.sum(y_true_f) + K.sum(y_pred_f))
    return score


def iou(y_true, y_pred, return_raw=False):
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3]) - intersection
    if return_raw:
        return intersection, union
    return intersection / union


def auc_roc(y_true, y_pred):
    value, update_op = tf.metrics.auc(y_true, y_pred)
    metric_vars = [i for i in tf.local_variables() if 'auc_roc' in i.name.split('/')[1]]

    for v in metric_vars:
        tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)

    with tf.control_dependencies([update_op]):
        value = tf.identity(value)

    return value


def map_iou(y_true, y_pred):
    y_pred = K.round(y_pred)
    ious = iou(y_true, y_pred)

    def precision_at(threshold):
        mask = K.greater(ious, threshold)
        return mask

    precisions = [precision_at(t) for t in np.arange(0.5, 1.0, 0.05)]
    return K.mean(K.stack(precisions, axis=1))


def slow_map_iou(y_true, y_pred):

    y_true_ = tf.cast(tf.round(y_true), tf.bool)
    y_pred_ = tf.cast(tf.round(y_pred), tf.bool)

    # Flatten
    y_true_ = tf.reshape(y_true_, shape=[tf.shape(y_true_)[0], -1])
    y_pred_ = tf.reshape(y_pred_, shape=[tf.shape(y_pred_)[0], -1])
    threasholds_iou = tf.constant(np.arange(0.5, 1.0, 0.05), dtype=tf.float32)

    def _mean_score(y):
        """Calculate score per image"""
        y0, y1 = y[0], y[1]
        total_cm = tf.confusion_matrix(y0, y1, num_classes=2)
        # total_cm = tf.Print(total_cm, [total_cm])
        sum_over_row = tf.to_float(tf.reduce_sum(total_cm, 0))
        sum_over_col = tf.to_float(tf.reduce_sum(total_cm, 1))
        cm_diag = tf.to_float(tf.diag_part(total_cm))
        denominator = sum_over_row + sum_over_col - cm_diag
        denominator = tf.where(tf.greater(denominator, 0), denominator, tf.ones_like(denominator))
        iou_val = tf.div(cm_diag, denominator)
        # iou_val = tf.Print(iou_val, [iou_val])
        iou_fg = iou_val[1]
        greater = tf.greater(iou_fg, threasholds_iou)
        score_per_image = tf.reduce_mean(tf.cast(greater, tf.float32))
        score_per_image = tf.where(
            tf.logical_and(
                tf.equal(tf.reduce_any(y0), False), tf.equal(tf.reduce_any(y1), False)),
            1., score_per_image)
        return score_per_image

    elems = (y_true_, y_pred_)
    scores_per_image = tf.map_fn(_mean_score, elems, dtype=tf.float32)
    return tf.reduce_mean(scores_per_image)


def get_map_iou_at(map_func=map_iou):
    """Mean average Precision IOU at diff threshols """

    def map_iou_at(y_true, y_pred):
        map_ious = []
        for t in np.arange(0, 1.0, 0.05):
            y_pred_cast = K.cast(y_pred > t,  dtype='float32')
            map_ious.append(map_func(y_true, y_pred_cast))
        return K.mean(K.stack(map_ious))

    return map_iou_at


##########
# Losses #
##########


def get_map_loss(which='map_iou', at=False, log=False):

    map_ious_maps = {
        'map_iou': map_iou,
        'slow_map_iou': slow_map_iou,
    }
    # map_func = which if callable(which) else map_ious_maps[which]
    map_func = map_ious_maps[which]
    map_func = get_map_iou_at(map_func) if at else map_func

    def map_loss(y_true, y_pred):
        if log:
            return binary_crossentropy(y_true, y_pred) + K.log(map_func(y_true, y_pred))
        return binary_crossentropy(y_true, y_pred) + (1. - map_func(y_true, y_pred))

    return map_loss


def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = y_true_f * y_pred_f
    score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1. - score


def get_bce_dice_loss(log=False):

    def bce_dice_loss(y_true, y_pred):
        if log:
            return binary_crossentropy(y_true, y_pred) - K.log(1. - dice_loss(y_true, y_pred))
        return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

    return bce_dice_loss


def weighted_bce_loss(y_true, y_pred, weight):
    epsilon = 1e-7
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
    logit_y_pred = K.log(y_pred / (1. - y_pred))
    loss = weight * (logit_y_pred * (1. - y_true) +
                     K.log(1. + K.exp(-K.abs(logit_y_pred))) + K.maximum(-logit_y_pred, 0.))
    return K.sum(loss) / K.sum(weight)


def weighted_dice_loss(y_true, y_pred, weight):
    smooth = 1.
    w, m1, m2 = weight, y_true, y_pred
    intersection = (m1 * m2)
    score = (2. * K.sum(w * intersection) + smooth) / (K.sum(w * m1) + K.sum(w * m2) + smooth)
    loss = 1. - K.sum(score)
    return loss


def weighted_bce_dice_loss(y_true, y_pred):
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    # if we want to get same size of output, kernel size must be odd
    averaged_mask = K.pool2d(
        y_true, pool_size=(50, 50), strides=(1, 1), padding='same', pool_mode='avg')
    weight = K.ones_like(averaged_mask)
    w0 = K.sum(weight)
    weight = 5. * K.exp(-5. * K.abs(averaged_mask - 0.5))
    w1 = K.sum(weight)
    weight *= (w0 / w1)
    loss = weighted_bce_loss(y_true, y_pred, weight) + dice_loss(y_true, y_pred)
    return loss


def weight_loss_wrapper(loss_func, protocol=0, pos_weight=.2, neg_weight=.8, pen_no_mask=False):
    """
    Idea:
    ---
    diff = sum of y_pred pixel - sum of y_true pixel
    (diff /= total_numbers of pixels) ** 2
    ---
    OR
    ---
    diff = sum of y_pred pixel - sum of y_true pixel
    diff /= total_numbers of pixels
    if diff > 0: # when the model is predicting too many pixels
        diff *= positive_weight
    else: # when the model is predicting fewer pixels
        diff *= negative_weight 
    diff **=2
    ---
    OR
    ---
    any of above combined with zero zero weight
    if sum of y_true pixel == 0 and sum of y_pred pixel > 0:
        diff *= (1 + (sum of y_pred pixel / total_numbers of pixels))
    """

    def calc_diff(y_true, y_pred):
        return K.sum(y_pred, axis=[1, 2, 3]) - K.sum(y_true, axis=[1, 2, 3])

    def protocol_0(y_true, y_pred, array=False):
        diff = calc_diff(y_true, y_pred)
        diff /= tf.cast(tf.reduce_prod(tf.shape(y_true)[1:]), tf.float32)
        diff = diff ** 2
        if array:
            return diff
        return K.mean(diff)

    def protocol_1(y_true, y_pred, array=False):
        diff = calc_diff(y_true, y_pred)
        diff /= tf.cast(tf.reduce_prod(tf.shape(y_true)[1:]), tf.float32)
        tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'dadff')
        v = tf.Variable(diff, name='dadff', dtype='float32', validate_shape=False)
        diff = tf.scatter_mul(tf.identity(v), tf.where(diff < 0), neg_weight)
        # diff = tf.scatter_mul(diff, tf.where(diff < 0), neg_weight)
        diff = tf.scatter_mul(K.variable(diff, ), tf.where(diff > 0), pos_weight)
        # diff = tf.scatter_mul(diff, tf.where(diff > 0), pos_weight)
        diff = diff ** 2
        if array:
            return diff
        return K.mean(diff)

    def penalize_no_mask(y_true, y_pred, loss):
        """Penalize model if ground truth contains no mask
        and prediction contains masks"""
        gt_sum = K.sum(y_true, axis=[1, 2, 3])
        pred_sum = K.sum(y_pred, axis=[1, 2, 3])
        gt_no_masks = tf.equal(gt_sum, 0.)
        pred_has_mask = tf.greater(pred_sum, 0.)
        should_not_have_mask = tf.cast(gt_no_masks, tf.float32) * tf.cast(pred_has_mask, tf.float32)
        pred_mask_perc = pred_sum / tf.cast(tf.reduce_prod(tf.shape(y_true)[1:]), tf.float32)
        no_mask_weight = tf.multiply(should_not_have_mask, pred_mask_perc, name='weight_mul') + 1
        return tf.multiply(loss, no_mask_weight, name='loss_mask_mul')

    protocol_map = {
        0: protocol_0,
        1: protocol_1,
    }

    def weight_loss(y_true, y_pred):
        if pen_no_mask:
            weight = protocol_map[protocol](y_true, y_pred, True)
            weight = penalize_no_mask(y_true, y_pred, weight)
        else:
            weight = protocol_map[protocol](y_true, y_pred)
        loss = tf.multiply(loss_func(y_true, y_pred), (weight + 1), name='loss_mul')
        return loss

    return weight_loss
