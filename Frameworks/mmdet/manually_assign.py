from mmdet.core.bbox.assigners import MaxIoUAssigner

gts = torch.tensor([[]]) #  K1 x 4, K1 is the ground truth bounding box number
dts = torch.tensor([[]]) #  K2 x 4, K2 is the detection truth bounding box number
labels = torch.tensor([]) # K1, K1 is the number of gt labels

assigner=MaxIoUAssigner(pos_iou_thr=0.5,neg_iou_thr=0.5,min_pos_iou=0.5,match_low_quality=False,ignore_iof_thr=-1)


''' 
  in re(class AssignResult):
  num_gts is number of ground truths;
  gt_inds is the inds of gt_bboxes assigned with bboxes, in gt_inds 0 means not assigned and 1 means the first in gt_bboxes;
  max_overlaps is the max IOU of bboxes between bboxes and gt_bboxes;
  labels just like gt_inds.
'''

re = assigner.assign(bboxes=dts,gt_bboxes=gts,gt_bboxes_ignore=None,gt_lables=labels)



  
