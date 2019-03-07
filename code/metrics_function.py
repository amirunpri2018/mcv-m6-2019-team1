import utils as u


def compute_metrics(gt, img_shape, noise_size=5, noise_position=5, create_bbox_proba=0.5, destroy_bbox_proba=0.5, k=10):

    """
    1. Add noise to ground truth Bounding Boxes.
    2.Compute Fscore, IoU, Map of two lists of Bounding Boxes.

    :param gt: list of GT bounding boxes
    :param img_shape: Original image shape
    :param noise_size: Change bbox size param
    :param noise_position: Increase bbox size param
    :param destroy_bbox_proba: Proba of destroying Bboxes
    :param create_bbox_proba: Proba of creating Bboxes
    :param k: Map at k
    :return: Noisy Bboxes, Fscore, IoU, MaP
    """

    # Add noise to GT depending on noise parameter
    bboxes = u.add_noise_to_bboxes(gt, img_shape,
                                   noise_size=True,
                                   noise_size_factor=noise_size,
                                   noise_position=True,
                                   noise_position_factor=noise_position)

    # Randomly create and destroy bounding boxes depending
    # on probability parameter
    bboxes = u.create_bboxes(bboxes, img_shape, prob=create_bbox_proba)
    bboxes = u.destroy_bboxes(bboxes, prob=destroy_bbox_proba)


    bboxTP, bboxFN, bboxFP = evalf.performance_accumulation_window(bboxes, gt)

    """
    Compute F-score of GT against modified bboxes PER FRAME NUMBER
    """
    # ToDo: Add dependency on frame number

    fscore = u.fscore(bboxTP, bboxFN, bboxFP)

    """
    Compute IoU of GT against modified Bboxes PER FRAME NUMBER:
    """
    iou = list()

    for b, box in enumerate(gt):
        iou.append(u.bbox_iou(bboxes[b], gt[b]))

    """
    Compute mAP of GT against modified bboxes PER FRAME NUMBER:
    """
    map = u.mapk(bboxes, gt, k)

    return (bboxes, fscore, iou, map)
