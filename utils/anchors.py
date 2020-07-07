import keras
import numpy as np

from utils.config import Config

config = Config()


def generate_anchors(sizes=None, ratios=None):
    ''' 生成大小不同的先验框的边长
    一共九个
    [[ -64.,  -64.,   64.,   64.],
       [-128., -128.,  128.,  128.],
       [-256., -256.,  256.,  256.],
       [ -64., -128.,   64.,  128.],
       [-128., -256.,  128.,  256.],
       [-256., -512.,  256.,  512.],
       [-128.,  -64.,  128.,   64.],
       [-256., -128.,  256.,  128.],
       [-512., -256.,  512.,  256.]]
    '''
    if sizes is None:
        sizes = config.anchor_box_scales  # [128, 256, 512]

    if ratios is None:
        ratios =  config.anchor_box_ratios #  [[1, 1], [1, 2], [2, 1]]
    # 框的数目
    num_anchors = len(sizes) * len(ratios) # 3*3
    # 放置框
    anchors = np.zeros((num_anchors, 4))  #[9,4]

    anchors[:, 2:] = np.tile(sizes, (2, len(ratios))).T  # 把size复制成[2,3]
    
    for i in range(len(ratios)):
        anchors[3*i:3*i+3, 2] = anchors[3*i:3*i+3, 2]*ratios[i][0]
        anchors[3*i:3*i+3, 3] = anchors[3*i:3*i+3, 3]*ratios[i][1]
    

    anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
    anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T
    return anchors

def shift(shape, anchors, stride=config.rpn_stride):
    ''' 生成网格中心点 stride=16，根据边长和中心点生成框 What\How \Why
    [0,1,2,3,....37]
    [0.5,1.5,2.5,....,37.5]
    [0.5,1.5,2.5,....,37.5]*stride
    '''

    shift_x = (np.arange(0, shape[0], dtype=keras.backend.floatx()) + 0.5) * stride
    shift_y = (np.arange(0, shape[1], dtype=keras.backend.floatx()) + 0.5) * stride

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    shift_x = np.reshape(shift_x, [-1])
    shift_y = np.reshape(shift_y, [-1])

    shifts = np.stack([
        shift_x,
        shift_y,
        shift_x,
        shift_y
    ], axis=0)

    shifts            = np.transpose(shifts)
    number_of_anchors = np.shape(anchors)[0]

    k = np.shape(shifts)[0]
    # 下面两步操作得到框的左上角和右下角坐标
    shifted_anchors = np.reshape(anchors, [1, number_of_anchors, 4]) + np.array(np.reshape(shifts, [k, 1, 4]), keras.backend.floatx())
    shifted_anchors = np.reshape(shifted_anchors, [k * number_of_anchors, 4])
    return shifted_anchors

def get_anchors(shape,width,height):
    ''' 获得先验框
    shape： featuremap.shape ,img.width,img.height '''
    # 1 生成框边长
    anchors = generate_anchors()
    # 2 边长和中心点结合，每个像素点生成框
    network_anchors = shift(shape,anchors)
    # 3 缩放框在0，1之间
    network_anchors[:,0] = network_anchors[:,0]/width
    network_anchors[:,1] = network_anchors[:,1]/height
    network_anchors[:,2] = network_anchors[:,2]/width
    network_anchors[:,3] = network_anchors[:,3]/height
    network_anchors = np.clip(network_anchors,0,1)
    return network_anchors

if __name__=='__main__':
    network_anchors=get_anchors([38,38],600,600)
    print(len(network_anchors))
    print(network_anchors[1300:1310])