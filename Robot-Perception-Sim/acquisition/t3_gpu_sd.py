"""CUDA linear-HDR grayscale, biased SD and signed TD for the T3 stereo eyes."""
import warp as wp

@wp.kernel
def images(src:wp.array3d(dtype=wp.float16),rgb:wp.array3d(dtype=wp.uint8),gray:wp.array2d(dtype=wp.uint8)):
    y,x=wp.tid()
    r=float(src[y,x,0]);g=float(src[y,x,1]);b=float(src[y,x,2])
    luminance=r*0.2126+g*0.7152+b*0.0722
    gray[y,x]=wp.uint8(wp.rint(wp.clamp(luminance,0.,1.)*255.))
    for c in range(3):
        v=wp.clamp(float(src[y,x,c]),0.,1.)
        z=12.92*v
        if v>0.0031308:z=1.055*wp.pow(v,1./2.4)-0.055
        rgb[y,x,c]=wp.uint8(wp.rint(z*255.))

@wp.kernel
def differences(gray:wp.array2d(dtype=wp.uint8),out:wp.array3d(dtype=wp.int16)):
    y,x=wp.tid()
    u=0;r=0
    if y>0:u=int(gray[y,x])-int(gray[y-1,x])
    if x<gray.shape[1]-1:r=int(gray[y,x])-int(gray[y,x+1])
    out[0, y, x] = wp.int16(wp.trunc(float(u) * 0.5) + 128.0)
    out[1, y, x] = wp.int16(wp.trunc(float(r) * 0.5) + 128.0)


@wp.kernel
def temporal_difference(gray: wp.array2d(dtype=wp.uint8), previous: wp.array2d(dtype=wp.uint8),
                        out: wp.array2d(dtype=wp.int8), has_previous: bool):
    y, x = wp.tid()
    current = gray[y, x]
    delta = int(0)
    if has_previous:
        delta = int(current) - int(previous[y, x])
    out[y, x] = wp.int8(wp.trunc(float(delta) * 0.5))
    previous[y, x] = current


class TemporalDifference:
    """History for one eye, advanced once per delivered sensor sample."""

    def __init__(self, gray):
        self.previous = wp.empty_like(gray)
        self.output = wp.empty(gray.shape, dtype=wp.int8, device=gray.device)
        self.last_index = None

    def sample(self, gray, index):
        consecutive = self.last_index is not None and index == self.last_index + 1
        wp.launch(temporal_difference, gray.shape,
                  inputs=[gray, self.previous, self.output, consecutive], device=gray.device)
        self.last_index = index
        return self.output
