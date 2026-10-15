# src/rngd/vision/layout_qa.py
class PixelToFeet:
    def __init__(self, geom_sidecar: dict):
        (x0_ft, y0_ft), (x1_ft, y1_ft) = geom_sidecar["ref_ft"]
        (x0_px, y0_px), (x1_px, y1_px) = geom_sidecar["ref_px"]
        self.sx = (x1_ft - x0_ft) / (x1_px - x0_px)
        self.sy = (y1_ft - y0_ft) / (y1_px - y0_px)
        self.x0_ft, self.x0_px = x0_ft, x0_px
        self.y0_ft, self.y0_px = y0_ft, y0_px

    def to_ft(self, x_px: float, y_px: float) -> tuple[float, float]:
        x_ft = self.x0_ft + (x_px - self.x0_px) * self.sx
        y_ft = self.y0_ft + (y_px - self.y0_px) * self.sy
        return x_ft, y_ft
