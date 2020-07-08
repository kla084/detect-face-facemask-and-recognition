import cv2
import tensorflow as tf
from tensorflow.python.platform import gfile
import numpy as np
import time
import math

def model_restore_from_pb(pb_path,node_dict):
    config = tf.ConfigProto(log_device_placement=True,
                            allow_soft_placement=True,
                            )
    config.gpu_options.allow_growth = True

    sess = tf.Session(config=config)
    with gfile.FastGFile(pb_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')  # import calculation chart
    sess.run(tf.global_variables_initializer())
    for key,value in node_dict.items():
        node = sess.graph.get_tensor_by_name(value)
        node_dict[key] = node
    return sess,node_dict

cap = cv2.VideoCapture(0) # input from camera
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)

def generate_anchors(feature_map_sizes, anchor_sizes, anchor_ratios, offset=0.5):

    anchor_bboxes = []
    for idx, feature_size in enumerate(feature_map_sizes):
        cx = (np.linspace(0, feature_size[0] - 1, feature_size[0]) + 0.5) / feature_size[0]
        cy = (np.linspace(0, feature_size[1] - 1, feature_size[1]) + 0.5) / feature_size[1]
        cx_grid, cy_grid = np.meshgrid(cx, cy)
        cx_grid_expend = np.expand_dims(cx_grid, axis=-1)
        cy_grid_expend = np.expand_dims(cy_grid, axis=-1)
        center = np.concatenate((cx_grid_expend, cy_grid_expend), axis=-1)

        num_anchors = len(anchor_sizes[idx]) +  len(anchor_ratios[idx]) - 1
        center_tiled = np.tile(center, (1, 1, 2* num_anchors))
        anchor_width_heights = []

        # different scales with the first aspect ratio
        for scale in anchor_sizes[idx]:
            ratio = anchor_ratios[idx][0] # select the first ratio
            width = scale * np.sqrt(ratio)
            height = scale / np.sqrt(ratio)
            anchor_width_heights.extend([-width / 2.0, -height / 2.0, width / 2.0, height / 2.0])

        # the first scale, with different aspect ratios (except the first one)
        for ratio in anchor_ratios[idx][1:]:
            s1 = anchor_sizes[idx][0] # select the first scale
            width = s1 * np.sqrt(ratio)
            height = s1 / np.sqrt(ratio)
            anchor_width_heights.extend([-width / 2.0, -height / 2.0, width / 2.0, height / 2.0])

        bbox_coords = center_tiled + np.array(anchor_width_heights)
        bbox_coords_reshape = bbox_coords.reshape((-1, 4))
        anchor_bboxes.append(bbox_coords_reshape)
    anchor_bboxes = np.concatenate(anchor_bboxes, axis=0)
    return anchor_bboxes

def decode_bbox(anchors, raw_outputs, variances=[0.1, 0.1, 0.2, 0.2]):

    #Decode the actual bbox according to the anchors.
    #the anchor value order is:[xmin,ymin, xmax, ymax]

    anchor_centers_x = (anchors[:, :, 0:1] + anchors[:, :, 2:3]) / 2
    anchor_centers_y = (anchors[:, :, 1:2] + anchors[:, :, 3:]) / 2
    anchors_w = anchors[:, :, 2:3] - anchors[:, :, 0:1]
    anchors_h = anchors[:, :, 3:] - anchors[:, :, 1:2]
    raw_outputs_rescale = raw_outputs * np.array(variances)
    predict_center_x = raw_outputs_rescale[:, :, 0:1] * anchors_w + anchor_centers_x
    predict_center_y = raw_outputs_rescale[:, :, 1:2] * anchors_h + anchor_centers_y
    predict_w = np.exp(raw_outputs_rescale[:, :, 2:3]) * anchors_w
    predict_h = np.exp(raw_outputs_rescale[:, :, 3:]) * anchors_h
    predict_xmin = predict_center_x - predict_w / 2
    predict_ymin = predict_center_y - predict_h / 2
    predict_xmax = predict_center_x + predict_w / 2
    predict_ymax = predict_center_y + predict_h / 2
    predict_bbox = np.concatenate([predict_xmin, predict_ymin, predict_xmax, predict_ymax], axis=-1)
    return predict_bbox

def single_class_non_max_suppression(bboxes, confidences, conf_thresh=0.2, iou_thresh=0.5, keep_top_k=-1):
    '''
    do nms on single class.
    Hint: for the specific class, given the bbox and its confidence,
    1) sort the bbox according to the confidence from top to down, we call this a set
    2) select the bbox with the highest confidence, remove it from set, and do IOU calculate with the rest bbox
    3) remove the bbox whose IOU is higher than the iou_thresh from the set,
    4) loop step 2 and 3, util the set is empty.
    '''

    if len(bboxes) == 0: return []

    conf_keep_idx = np.where(confidences > conf_thresh)[0]

    bboxes = bboxes[conf_keep_idx]
    confidences = confidences[conf_keep_idx]

    pick = []
    xmin = bboxes[:, 0]
    ymin = bboxes[:, 1]
    xmax = bboxes[:, 2]
    ymax = bboxes[:, 3]

    area = (xmax - xmin + 1e-3) * (ymax - ymin + 1e-3)
    idxs = np.argsort(confidences)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # keep top k
        if keep_top_k != -1:
            if len(pick) >= keep_top_k:
                break

        overlap_xmin = np.maximum(xmin[i], xmin[idxs[:last]])
        overlap_ymin = np.maximum(ymin[i], ymin[idxs[:last]])
        overlap_xmax = np.minimum(xmax[i], xmax[idxs[:last]])
        overlap_ymax = np.minimum(ymax[i], ymax[idxs[:last]])
        overlap_w = np.maximum(0, overlap_xmax - overlap_xmin)
        overlap_h = np.maximum(0, overlap_ymax - overlap_ymin)
        overlap_area = overlap_w * overlap_h
        overlap_ratio = overlap_area / (area[idxs[:last]] + area[i] - overlap_area)

        need_to_be_deleted_idx = np.concatenate(([last], np.where(overlap_ratio > iou_thresh)[0]))
        idxs = np.delete(idxs, need_to_be_deleted_idx)

    # if the number of final bboxes is less than keep_top_k, we need to pad it.
    return conf_keep_idx[pick]


pb_path = "mask_model.pb"
node_dict = {'input':'data_1:0',
            'detection_bboxes':'loc_branch_concat_1/concat:0',
            'detection_scores':'cls_branch_concat_1/concat:0'}
conf_thresh = 0.5
iou_thresh = 0.4

FPS = "0"
frame_count = 0
#anchors config
feature_map_sizes = [[33, 33], [17, 17], [9, 9], [5, 5], [3, 3]]
anchor_sizes = [[0.04, 0.056], [0.08, 0.11], [0.16, 0.22], [0.32, 0.45], [0.64, 0.72]]
anchor_ratios = [[1, 0.62, 0.42]] * 5
id2class = {0: 'WEAR MASK', 1: 'NO MASK'} #0: has mask, 1:no mask

#video streaming init

#model init

#generate anchors
anchors = generate_anchors(feature_map_sizes, anchor_sizes, anchor_ratios)
# for inference , the batch size is 1, the model output shape is [1, N, 4],
# so we expand dim for anchors to [1, anchor_num, 4]
anchors_exp = np.expand_dims(anchors, axis=0)

#model restore from pb file
sess,node_dict = model_restore_from_pb(pb_path, node_dict)
tf_input = node_dict['input']
model_shape = tf_input.shape #[N,H,W,C]
print("model_shape -> ", model_shape)
detection_bboxes = node_dict['detection_bboxes']
detection_scores = node_dict['detection_scores']

while (cap.isOpened()):

    #get image
    ret, img = cap.read()

    if ret:
        #image processing
        img_resized = cv2.resize(img, (model_shape[2], model_shape[1]))
        img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_resized = img_resized.astype('float32')
        img_resized /= 255

        #mask detection
        y_bboxes_output, y_cls_output = sess.run([detection_bboxes, detection_scores],
                                                feed_dict={tf_input: np.expand_dims(img_resized, axis=0)})

        #remove the batch dimension, for batch is always 1 for inference.
        y_bboxes = decode_bbox(anchors_exp, y_bboxes_output)[0]
        y_cls = y_cls_output[0]
        # To speed up, do single class NMS, not multiple classes NMS.
        bbox_max_scores = np.max(y_cls, axis=1)
        bbox_max_score_classes = np.argmax(y_cls, axis=1)

        # keep_idx is the alive bounding box after nms.
        keep_idxs = single_class_non_max_suppression(y_bboxes,
                                                    bbox_max_scores,
                                                    conf_thresh=conf_thresh,
                                                    iou_thresh=iou_thresh,
                                                    )
        #draw bounding box
        for idx in keep_idxs:
            conf = float(bbox_max_scores[idx])
            class_id = bbox_max_score_classes[idx]
            bbox = y_bboxes[idx]
            # clip the coordinate, avoid the value exceed the image boundary.
            xmin = max(0, int(bbox[0] * width))
            ymin = max(0, int(bbox[1] * height))
            xmax = min(int(bbox[2] * width), width)
            ymax = min(int(bbox[3] * height), height)

            
            #Calculation of distance to camera
            distancei = (2 * 3.14 * 180) / (int(ymin) + int(ymax) * 360) * 1000 + 3
            #print(distancei)
            distance = distancei * 2.54
            distance = math.floor(distance)
            cv2.putText(img, str(distance), ( 20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0))
            

            if class_id == 0: #if detected face has mask
                color = (0, 255, 0)
                
            else:
                color = (0, 0, 255)

            cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2) # face -> rectangle
            cv2.putText(img, "%s: %.2f" % (id2class[class_id], conf), (int(xmin), int(ymax) +15),cv2.FONT_HERSHEY_SIMPLEX, 0.5, color,2) # text for detected faces( have mask or not )

        if frame_count == 0:
            t_start = time.time()
        frame_count += 1
        if frame_count >= 10:
            FPS = "FPS=%1f" % (10 / (time.time() - t_start))
            frame_count = 0
        #cv2.putText(影像, 文字, 座標, 字型, 大小, 顏色, 線條寬度, 線條種類)
        cv2.putText(img, FPS, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)


        # number of people have not mask

        '''
        limit = 5 # max people limit
        #limit depends on camera view angle
        #can be 10% of the camera field of view
        if keep_idxs.shape[0]>= limit: # if total detected faces > limit
            color_limit= (0,0,255) #if over the limit color is red
            cv2.putText(img, "SOSYAL MESAFE SINIRI !" , (16, 130), cv2.FONT_HERSHEY_SIMPLEX, 2,
                        (0, 0, 255), 3) # social distance limit
        else: color_limit= (0,0,0) #else black
        # cv2.putText(img, "TOPLAM: "+ str(keep_idxs.shape[0]), (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
        #            color_sinir, 2)  #number of total faces
        '''

        #image display
        cv2.imshow("mask detection", img)

        #image writing

        if cv2.waitKey(1) & 0xFF == ord('q'): # pressed q --> exit
            break
    else:
        print("get image error")
        break

cap.release()
cv2.destroyAllWindows()
