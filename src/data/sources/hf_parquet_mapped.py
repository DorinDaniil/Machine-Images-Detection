"""`hf_parquet` whose generator comes from a mapping instead of a raw column value.

Same streaming download as generic.HFParquetSource (stops at max_items), plus
  generator_by_label: {1: flux.1, 2: ldm-inpaint}   raw label value -> generator slug
  generator_map: {"3": sdxl}                        raw generator_column value -> slug
Useful when one label value means "fully generated" and another "inpainted", or when the
generator column is a ClassLabel (streamed as an integer).

    kind: custom
    adapter: hf_parquet_mapped
    args:
      repo: saberzl/SID_Set
      split: validation
      label_column: label
      label_map: {"2": 1}
      generator_by_label: {"1": flux.1, "2": ldm-inpaint}
      mask_column: mask
      content_type: photo
      manifest_split: test
"""

from .base import register
from .generic import HFParquetSource


@register
class HFParquetMapped(HFParquetSource):
    name = "hf_parquet_mapped"

    def __init__(self, generator_by_label: dict | None = None, generator_map: dict | None = None, **kwargs):
        super().__init__(name=self.name, **kwargs)
        self.generator_by_label = {str(k): v for k, v in (generator_by_label or {}).items()}
        self.generator_map = {str(k).lower(): v for k, v in (generator_map or {}).items()}

    def _generator(self, row: dict) -> str:
        if self.generator_by_label and self.label_column:
            generator = self.generator_by_label.get(str(row[self.label_column]))
            if generator is not None:
                return generator
        raw = super()._generator(row)
        return self.generator_map.get(raw, raw)
