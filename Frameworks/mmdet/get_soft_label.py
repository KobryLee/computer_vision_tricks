# get soft label of an image
from mmdet.apis import init_detector
from mmdet.core import multiclass_nms
from mmdet.datasets.pipelines import Compose
import torch


# config file of soft
model = init_detector("configs/soft_teacher/soft_teacher_faster_rcnn_r50_caffe_fpn_coco_180k.py", "pretrain/soft_teacher.pth", device='cuda:1')

file="data/coco/images/test2017/000000000069.jpg"
test_pipeline = Compose(model.cfg.test_pipeline)
data = dict(img_info=dict(filename=file), img_prefix=None)
data=test_pipeline(data)


'''
    the img goal dimension is [N,3,H,W], N is the number of imgs
    
    if multiple images with same H and W, shape is [2,3,H,W]
        img = torch.cat(data1['img'][0].unsqeeze(dim=0),
                        data2['img'][0].unsqeeze(dim=0),
                        dim=0)
    
'''

img=data['img'][0].unsqueeze(dim=0).to("cuda:1")

with torch.no_grad():
    model.eval()
    feat= model.extract_feat(img)
    rpn_out = list(model.rpn_head(feat))
    # print(len(rpn_out[0][0][0]))
    proposal_list = model.rpn_head.get_bboxes(
        *rpn_out, [data['img_metas'][0].data], cfg=model.test_cfg.rpn
    )
    # rescale will only make x/scale_factor[0], y/scale_factor[1]
    # if rcnn_test_cfg is not None, porposal_list will be [N,K,5](x1,y1,x2,y2,score), proposal_label_list will be [N,K]
    # if rcnn_test_cfg is None, proposal_list if [N,K,class_num*4](class_num do not contain background class), 
    #                     proposal_label_list is [N,K,class_num+1] is the soft_label and background always be ind=class_num
    proposal_list, proposal_label_list = model.roi_head.simple_test_bboxes(
        feat, [data['img_metas'][0].data], proposal_list,rcnn_test_cfg = None ,rescale=False
    )
    # bbox is [N,K,5], soft is [N,K,class_num+1] and label is [N,K]
    bbox=[]
    label=[]
    inds=[]
    for img_id in range(img.size(0)):
        _bbox,_label,_inds=multiclass_nms(proposal_list[0],proposal_label_list[0],model.test_cfg.rcnn.score_thr, model.test_cfg.rcnn.nms,
                                                    model.test_cfg.rcnn.max_per_img,return_inds=True)
        bbox.append(_bbox)
        label.append(_label)
        inds.append(_inds)
        
    soft=[]
    # ind//class_num because soft_label is flattened to [N*K*class_num], and bbox is flattend to [N*K*class_num,4]
    for img_id in range(img.size(0)):
        temp=[]
        for ind in inds[img_id]:
            temp.append(proposal_label_list[img_id][ind//class_num])
        soft.append(temp)
