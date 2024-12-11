from pcdet.models.roi_heads.partA2_head import PartA2FCHead
from pcdet.models.roi_heads.pointrcnn_head import PointRCNNHead
from pcdet.models.roi_heads.pvrcnn_head import PVRCNNHead
from pcdet.models.roi_heads.roi_head_template import RoIHeadTemplate
from pcdet.models.roi_heads.second_head import SECONDHead

__all__ = {
    "RoIHeadTemplate": RoIHeadTemplate,
    "PartA2FCHead": PartA2FCHead,
    "PVRCNNHead": PVRCNNHead,
    "SECONDHead": SECONDHead,
    "PointRCNNHead": PointRCNNHead,
}
