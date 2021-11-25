# transform det bboxes under a data aug process to another data aug process
# aug process contain: flip, resize and so on, transform the size and position to a specific size and position


''' transform matrix [ 3 x 3 ]
     projective/perspective transformation：
    
     [x'       [[ a , b , c ],       [x,
      y'    =   [ e , f , g ],   *    y,
      z']       [ 0 , 0 , 1 ]]        z]
     
     z is for changing the img position, for example c is will be added to x, make image left or right
      
     so aug1_matrix=A
        aug2_matrix=B
        coor1 = A * coor_ori
        coor2 = B * coor_ori
        
     to get bboxes with aug1 to bboxes with aug2
     
     A^-1 * coor1 = B^-1 * coor2
     
     coor2 = B * A^-1 * coor1
     
     
     A is transform matrix of aug1 [N,3,3], N is the number of imgs
     B is transform matrix of aug2 [N,3,3], N is the number of imgs
     bboxes_1 is [N,M,4], M is the number of proposals in an image
     aug2_img_shape is [N,3] 3 is (H,W,C) can get from img_metas
'''


M = get_transform_matrix[
  A,
  B
]

bboxes_2 = transform_bbox[
  bboxes_1,
  M,
  aug2_img_shape
]
# @ is matrix multiply similar to torch.mm(a,b), while torch.mul(a,b) is element-wise
def get_trans_mat(a, b):
  return [bt @ at.inverse() for bt, at in zip(b, a)]


def transform_bboxes(bbox, M, out_shape):
    if isinstance(bbox, Sequence):
        assert len(bbox) == len(M)
        return [
            Transform2D.transform_bboxes(b, m, o)
            for b, m, o in zip(bbox, M, out_shape)
        ]
    else:
        if bbox.shape[0] == 0:
            return bbox
        score = None
        if bbox.shape[1] > 4:
            score = bbox[:, 4:]
        points = bbox2points(bbox[:, :4])
        points = torch.cat(
            [points, points.new_ones(points.shape[0], 1)], dim=1
        )  # n,3
        points = torch.matmul(M, points.t()).t()
        points = points[:, :2] / points[:, 2:3]
        bbox = points2bbox(points, out_shape[1], out_shape[0])
        if score is not None:
            return torch.cat([bbox, score], dim=1)
    return bbox
  
def bbox2points(box):
    min_x, min_y, max_x, max_y = torch.split(box[:, :4], [1, 1, 1, 1], dim=1)

    return torch.cat(
        [min_x, min_y, max_x, min_y, max_x, max_y, min_x, max_y], dim=1
    ).reshape(
        -1, 2
    )  # n*4,2


def points2bbox(point, max_w, max_h):
    point = point.reshape(-1, 4, 2)
    if point.size()[0] > 0:
        min_xy = point.min(dim=1)[0]
        max_xy = point.max(dim=1)[0]
        xmin = min_xy[:, 0].clamp(min=0, max=max_w)
        ymin = min_xy[:, 1].clamp(min=0, max=max_h)
        xmax = max_xy[:, 0].clamp(min=0, max=max_w)
        ymax = max_xy[:, 1].clamp(min=0, max=max_h)
        min_xy = torch.stack([xmin, ymin], dim=1)
        max_xy = torch.stack([xmax, ymax], dim=1)
        return torch.cat([min_xy, max_xy], dim=1)  # n,4
    else:
        return point.new_zeros(0, 4)



