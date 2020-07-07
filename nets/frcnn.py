from keras.layers import Conv2D, Input, TimeDistributed, Flatten, Dense, Reshape
from keras.models import Model

from nets.RoiPoolingConv import RoiPoolingConv
from nets.resnet import ResNet50, classifier_layers


def get_rpn(base_layers, num_anchors):
    # 1 base_layers进行3*3卷积
    x = Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='normal', name='rpn_conv1')(base_layers)
    # 2 得到每个框的置信度和框坐标
    x_class = Conv2D(num_anchors, (1, 1), activation='sigmoid', kernel_initializer='uniform', name='rpn_out_class')(x)
    x_regr = Conv2D(num_anchors * 4, (1, 1), activation='linear', kernel_initializer='zero', name='rpn_out_regress')(x)
    
    x_class = Reshape((-1,1),name="classification")(x_class)  # 如果包含物体，x_class的值接近1。
    x_regr = Reshape((-1,4),name="regression")(x_regr)
    return [x_class, x_regr, base_layers]

def get_classifier(base_layers, input_rois, num_rois, nb_classes=21, trainable=False):
    ''' roi --> cls+reg'''
    pooling_regions = 14
    input_shape = (num_rois, 14, 14, 1024)
    # base_layers[38,38,1024], input_rois[num_prior,4]  num_prior=32
    # 1 roiPooling，base_layers[38,38,1024], input_rois[-,4] ,pooling_regions=14, num_rois=32
    out_roi_pool = RoiPoolingConv(pooling_regions, num_rois)([base_layers, input_rois]) # input_rois是建议框
    # 2
    out = classifier_layers(out_roi_pool, input_shape=input_shape, trainable=True)
    out = TimeDistributed(Flatten())(out)
    out_class = TimeDistributed(Dense(nb_classes, activation='softmax', kernel_initializer='zero'), name='dense_class_{}'.format(nb_classes))(out)
    out_regr = TimeDistributed(Dense(4 * (nb_classes-1), activation='linear', kernel_initializer='zero'), name='dense_regress_{}'.format(nb_classes))(out)
    return [out_class, out_regr]  # 21, 20*4

def get_model(config,num_classes):
    ''' RPN网络输出 置信度+ 框回归 '''
    inputs = Input(shape=(None, None, 3))
    roi_input = Input(shape=(None, 4))
    base_layers = ResNet50(inputs)                # 共享特征层FeatureMap
    # 1 rpn网络
    num_anchors = len(config.anchor_box_scales) * len(config.anchor_box_ratios) # 9=3*3
    rpn = get_rpn(base_layers, num_anchors)       # rpn网络输出[x_class, x_regr, base_layers]
    model_rpn = Model(inputs, rpn[:2])  # ？？？
    # 2 fast rcnn网络
    classifier = get_classifier(base_layers, roi_input, config.num_rois, nb_classes=num_classes, trainable=True)
    model_classifier = Model([inputs, roi_input], classifier)
    # 3 综合
    model_all = Model([inputs, roi_input], rpn[:2]+classifier)
    return model_rpn,model_classifier,model_all

def get_predict_model(config,num_classes):
    inputs = Input(shape=(None, None, 3))
    roi_input = Input(shape=(None, 4))
    feature_map_input = Input(shape=(None,None,1024))

    base_layers = ResNet50(inputs)
    num_anchors = len(config.anchor_box_scales) * len(config.anchor_box_ratios)
    rpn = get_rpn(base_layers, num_anchors)
    model_rpn = Model(inputs, rpn)

    classifier = get_classifier(feature_map_input, roi_input, config.num_rois, nb_classes=num_classes, trainable=True)
    model_classifier_only = Model([feature_map_input, roi_input], classifier)

    return model_rpn,model_classifier_only