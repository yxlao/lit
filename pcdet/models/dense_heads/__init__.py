from pcdet.models.dense_heads.anchor_head_multi import AnchorHeadMulti
from pcdet.models.dense_heads.anchor_head_single import AnchorHeadSingle
from pcdet.models.dense_heads.anchor_head_template import AnchorHeadTemplate
from pcdet.models.dense_heads.point_head_box import PointHeadBox
from pcdet.models.dense_heads.point_head_simple import PointHeadSimple
from pcdet.models.dense_heads.point_intra_part_head import PointIntraPartOffsetHead

__all__ = {
    "AnchorHeadTemplate": AnchorHeadTemplate,
    "AnchorHeadSingle": AnchorHeadSingle,
    "PointIntraPartOffsetHead": PointIntraPartOffsetHead,
    "PointHeadSimple": PointHeadSimple,
    "PointHeadBox": PointHeadBox,
    "AnchorHeadMulti": AnchorHeadMulti,
}
