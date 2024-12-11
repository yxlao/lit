from pcdet.models.detectors.detector3d_template import Detector3DTemplate
from pcdet.models.detectors.PartA2_net import PartA2Net
from pcdet.models.detectors.point_rcnn import PointRCNN
from pcdet.models.detectors.pointpillar import PointPillar
from pcdet.models.detectors.pv_rcnn import PVRCNN
from pcdet.models.detectors.second_net import SECONDNet
from pcdet.models.detectors.second_net_iou import SECONDNetIoU

__all__ = {
    "Detector3DTemplate": Detector3DTemplate,
    "SECONDNet": SECONDNet,
    "PartA2Net": PartA2Net,
    "PVRCNN": PVRCNN,
    "PointPillar": PointPillar,
    "PointRCNN": PointRCNN,
    "SECONDNetIoU": SECONDNetIoU,
}


def build_detector(model_cfg, num_class, dataset):
    model = __all__[model_cfg.NAME](
        model_cfg=model_cfg, num_class=num_class, dataset=dataset
    )

    return model
