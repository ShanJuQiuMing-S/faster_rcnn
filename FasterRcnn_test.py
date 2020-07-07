主干网络提取特征-->特征进入RPN网络进行回归和分类，根据分类是否包含物体选择回顾结果-->
-->回归结果进入RoIpooling统一成统一大小--->上一步可以称之为抠图，抠图+特征进入Rcnn中得到最终的分类和回归结果。
''' 1 Faster-RCNN整体结构 Restnet50:In 600*600*3--->out 38*38*1024 (anchor=38*38*9)
    a. Backbone：Restnet50 In 600*600*3--->out feamap 38*38*1024
    b. RPN： b.1 conv3*3
                 b.1.1 conv1*1*9(k=9)
                 b.1.2 conv1*1*36(4*9)
             b.2 anchor+b.1.1+b.1.2--> proposal
    c. RCNN：In proposals+feamap-->抠图———>ROIPooling———>reg + cls    
'''

''' 2 Restnet50主干提取网络 resnet.py
a.ConvBlock : 用来调整featuremap尺寸
    conv1(input ,filter1)-->conv2(filter2)-->conv3(filter3) + conv(input)
b.IdentityBlock : 用来调整网络深度
    conv1(input ,filter1)-->conv2(filter2)-->conv3(filter3) + input
c.Restnet50：
    ZeroPadding2D、Conv2D、BatchNormalization、Activation、MaxPooling2D
    conv_block、identity_block*2
    conv_block、identity_block*3
    conv_block、identity_block*5
'''

''' 3  RPN建议框网络 nets/frcnn.py
3.1 get_rpn(base_layers, num_anchors)
    3.1.1 base_layers = ResNet50(inputs)
    3.1.2 rpn = get_rpn(base_layers, num_anchors)
        (1)'rpn_conv1' ： conv3*3 512
        (2)'rpn_out_class' : conv1*1  9
        (3)'rpn_out_regress' : conv1*1  4*9    
'''

''' 4 Anchor 先验框
4.1 生成框边长
    anchors = generate_anchors()
4.2 边长和中心点结合，每个像素点生成框
    network_anchors = shift(shape,anchors)
4.3 缩放框在0，1之间
'''

''' 5 Decode得到proposal  frcnn.py 
detect_image()
     # 1 图片resize，最短边600
        width,height = get_new_img_size() 
     # 2 图片预处理，归一化，导入rpn网络进行预测得到置信度和预测偏移值
        photo = preprocess_input()
        preds = self.model_rpn.predict()
            (1) ResNet50()
            (2) get_rpn()
            (3) get_classifier()
    # 3  生成先验框
        anchors = get_anchors()
    # 4 将预测结果进行解码--nms筛选得到建议框
        rpn_results = self.bbox_util.detection_out()
            (1) self.decode_boxes() 公式
            (2) 非极大值抑制
            (3) 抠图
    
'''

''' 5 ROI——Pooling nets/frcnn.py
get_classifier()
    RoiPoolingConv()   取框的xyhw,在feature上截取，resize.

'''

''' 6 Classifier_layers局部功用特征层网络
get_classifier()
    RoiPoolingConv()   取框的xyhw,在feature上截取，resize.
    classifier_layers()
    out_class = TimeDistributed(Dense())  21
    out_regr = TimeDistributed(Dense()) 80
判断置信度是否大于阈值\建议框类别\建议框坐标\ 建议框的调整参数
解码 - 中心
筛选出其中得分高于confidence的框
把box转化为小数
NMS
映射到原图\画框
'''

''' 7 预测框

'''

''' 8 Train
frcnn.py
"model_path": 'model_data/voc_weights.h5',  
"classes_path": 'model_data/voc_classes.txt'

train.py
  NUM_CLASSES = 21; 
base_net_weights
'''

''' 9 利用自己的数据训练

'''

''' 10 建议框网络训练

'''

''' 11 classifier

'''


