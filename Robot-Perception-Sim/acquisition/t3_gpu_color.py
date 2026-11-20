"""GPU exposure/sRGB conversion of Replicator FP16 HdrColor buffers."""
import warp as wp

@wp.kernel
def convert(src:wp.array3d(dtype=wp.float16),dst:wp.array3d(dtype=wp.uint8),meter:wp.array(dtype=float),scale:float):
    y,x=wp.tid()
    r=float(src[y,x,0])*scale;g=float(src[y,x,1])*scale;b=float(src[y,x,2])*scale
    if y>=120:
        wp.atomic_add(meter,0,.2126*r+.7152*g+.0722*b)
    for c in range(3):
        v=wp.clamp(float(src[y,x,c])*scale,0.,1.)
        z=12.92*v
        if v>.0031308:z=1.055*wp.pow(v,1./2.4)-.055
        dst[y,x,c]=wp.uint8(wp.rint(z*255.))
