# pySP
pySP is a Python-based raw converter that implements parts of a basic image signal processing pipeline.

## what is pySP for?
pySP is designed to simplify the less intuitive parts of raw image processing while still letting you interact with sensor data. It's **not a replacement for actual raw converters** but is useful for custom imaging pipelines.

pySP has the following features:
 - Raw bayer value decoding via rawpy (libraw)
 - Multiple debayering options for speed or quality
 - Color processing (camera white balancing, transforming to sRGB)
 - Bad photosite healing
 - HDR stacking in raw space
 - Shading (flat field) correction
 - Embedded per-channel distortion correction

Using debayering via rawpy, the following features are also available:

 - Denoising via "Fake Before Demosaicing Denoising"
 - Better demosaicing algorithms that improve detail and reduce fringing

pySP currently only supports RGBG sensors but because it uses libraw for raw decoding, most vendor-specific raw formats are supported natively.

## how do I install pySP?
After cloning pySP, install its requirements via pip, i.e.,

    pip install -r requirements.txt

Some performance-critical parts of pySP are written in Cython. Currently, compiler arguments are only correct under MSVC so can only be built for Windows x86-64. Compile this with the following:

    python setup.py build_ext --inplace

pySP is designed to be used as a submodule. Once cloned and built, it can be imported straightaway as a library.

## how do I use pySP?

### Converting a raw to an sRGB image
Raw processing transitions through different colorspaces. It starts at the sensor-level which are the values your camera sees through its filters. It is then debayered, white-balanced and transformed to XYZ where it can be shifted to any other space. pySP currently only supports sRGB and provides simple methods to develop from camera to linear RGB (and sRGB) where it can then be used like a normal image. The first step is debayering of which pySP offers two pathways.

Debayering via pySP's internal path uses the following algorithms at each quality setting:

 - Draft - Align and resample colors to quarter-res at pixel centers, bilinear scale back to original resolution
     - Fast with little fringing but very soft due to quarter resolution solve
 - Fast - Unimplemented, planned to trade off between Best/Draft
 - Best - Adaptive Homogeneity-Directed Demosaicing algorithm (Hirakawa, K. and Parks, T.W., 2005)
	 - Sharpens detail well with minimal zippering
	 - Slower than libraw for performance critical applications but supports HDR
	 - Implementation is naive and follows paper exactly; fringing is corrected using their color postprocessing method which can inadvertently remove small colored details
	 - Postprocessing can be adjusted with the `postprocess_stages` argument
     - Close match for other software using non-ML techniques

#### Debayering via pySP
    from pySP.const import QualityDemosaic
    from pySP.colorize import lin_srgb_to_srgb
    from pySP.image import RawRgbgDataFromRaw
    
    PATH_IMAGE_DNG = ...
    
    image = RawRgbgDataFromRaw(PATH_IMAGE_DNG)
    image_debayered_rgb = image.debayer(QualityDemosaic.Best).to_lin_srgb()
    debayered_srgb = lin_srgb_to_srgb(image_debayered_rgb)

#### Debayering via libraw (different algorithms, slower by default)

By default, rawpy is configured to apply noise reduction and will produce a slightly different tint to the image. Precision is otherwise the same so this is easy to correct. This method blocks access to raw Bayer data since pySP doesn't aim to replicate libraw's pipeline. Removing noise reduction will make debayering faster than pySP - if you don't require any pySP sensor-level features, use `rawpy` for image processing instead. This only exists as a means for bayer noise reduction in pySP and may be removed in the future.

    from pySP.colorize import lin_srgb_to_srgb
    from pySP.image import RawDebayerDataFromRaw
    
    PATH_IMAGE_DNG = ...
    
    image_debayered_rgb = RawDebayerDataFromRaw(PATH_IMAGE_DNG).to_lin_srgb()
    debayered_srgb = lin_srgb_to_srgb(image_debayered_rgb)

### Adjusting White Balance
**If you're happy with the colors in your image, you probably won't need to touch this.** Most cameras do a good job estimating the scene white.

pySP uses a multiplier-based white balancing system which tries to minimize unnatural color casts under a range of illuminants. It does this by blending pre-calibrated profiles inside the raw files. By default, it uses the camera's estimated illuminant to interpolate a profile which minimizes tint (cast) according to the Planckian locus. If your images contain no defined neutral multipliers or no color calibrations (which is extremely unlikely, especially if it's a DNG), pySP will fail to load your images.

#### Setting white balance from a neutral patch

    from pySP.colorize import lin_srgb_to_srgb
    from pySP.image import RawRgbgDataFromRaw
    
    PATH_IMAGE_DNG = ...
    
    image = RawRgbgDataFromRaw(PATH_IMAGE_DNG)
    image.cam_wb.update_by_reference(CAM_WHITE)

These can be found after debayering where the camera-space values of any neutral patch may be used to update color management for the raw image. Make sure to return to camera-space using `wb_undo()` before reading neutral values from the debayered image. Because debayering is sensitive to color, any adjustments to the neutral point require debayering to be done again to prevent strange tints entering the image.

#### Setting white balance from known temperature

    from pySP.colorize import lin_srgb_to_srgb
    from pySP.image import RawRgbgDataFromRaw
    
    PATH_IMAGE_DNG = ...
    
    image = RawRgbgDataFromRaw(PATH_IMAGE_DNG)
    image.cam_wb.update_by_temperature(ILL_TEMP, duv = ..., override_blend = ...)

When setting white point using temperature, it may be desirable to override blending behaviour to only use the closest camera calibration rather than attempt to interpolate one by setting `override_blend = 0`. Camera calibrations typically include 2 matrices from different series, one for daylight-series illuminants and the other for a standard illuminant. Blending calibrations between different illuminant series can create unnatural tints.

Under the hood pySP uses xyz color for defining illuminants. Temperature alone is insufficient to describe an illuminant so pySP exposes `duv` as a measure of the tint away from the Planckian locus - combining this with temperature gives an exact mapping for the desired illuminant. By default, pySP assumes the desired temperature is close to the daylight series (like other software) so will compute this for you when `duv = None`. There is a discontinuity at 4000K as there are no daylight series illuminants defined under 4000K so we snap back to the locus.

### Removing hot pixels in raws
Hot pixel removal works on raw Bayer data so is currently only compatible with pySP's debayering. Removal works in-place so the image can be used like normal after corrections (e.g., can then be debayered).

#### Consensus of single image (more false positives)

    from pySP.image import RawRgbgDataFromRaw
    from pySP.raw_bad_pixel_corr import find_erroneous_pixels_threshold, repair_bad_pixels
    
    PATH_IMAGE_DNG = ...
    
    image = RawRgbgDataFromRaw(PATH_IMAGE_DNG)
    repair_bad_pixels(image, find_erroneous_pixels_threshold(image))

#### Consensus of multiple images (more false negatives)

    from pySP.image import RawRgbgDataFromRaw
    from pySP.raw_bad_pixel_corr import find_erroneous_pixels_threshold, repair_bad_pixels, find_shared_pixels
    
    PATH_IMAGES_DNG = [...]
    
    images = [RawRgbgDataFromRaw(i) for i in PATH_IMAGES_DNG]
    bad_photosite_masks = find_shared_pixels([find_erroneous_pixels_threshold(i) for i in images], min_ratio=0.8)
    for image in images:
	    repair_bad_pixels(image, bad_photosite_masks)

### HDR stacking
High dynamic range stacking is provided as a way to reduce noise and expand dynamic range through exposure merging. All operations assume linear sensor response and image alignment and compute reference exposure through bundled metadata. Custom exposure can be set on a per-image basis by changing the `current_ev` member but this is only needed if sensor response is less linear or metadata is missing.

All stacking operations work inside camera-space although only sensor-space stacking can use raw Bayer data. pySP provides no highlight reconstruction so highlights retain perfect color saturation. Noise is reduced by biasing with reference to camera white balance and weighing in favor of closer exposures rather than amplifying noise present in further exposures.

#### Stacking in sensor-space (less artifacts, higher accuracy)
Sensor-space stacking works on photosite data so retains the complete dynamic range of each image with accurate clipping. Because this requires raw Bayer data, it is only compatible with pySP's debayering. The output of this method is a new Bayer image with photosite responses that can exceed 1.

    from pySP.image import RawRgbgDataFromRaw
    from pySP.raw_hdr import fuse_exposures_to_raw
    from pySP.raw_bad_pixel_corr import find_erroneous_pixels_threshold, repair_bad_pixels, find_shared_pixels
    
    PATH_IMAGES_DNG = [...]
    
    images = [RawRgbgDataFromRaw(i) for i in PATH_IMAGES_DNG]
    image_hdr, image_count = fuse_exposures_to_raw(images)

To render this image tonemapping is required. It is recommended that HDR images are exported and tonemapped in other software but a quick solution for debugging can be to use the Reinhardt tonemapper to produce an image that can be viewed correctly.

    from pySP.colorize import lin_srgb_to_srgb
    
    ...
    
    debayered_hdr_linear = image_hdr.debayer(QualityDemosaic.Best).to_lin_srgb()
	debayered_ldr = debayered_hdr_linear / (1 + debayered_hdr_linear)
    debayered_ldr_srgb = lin_srgb_to_srgb(debayered_ldr)

#### Stacking in camera-space (more artifacts, works with libraw)
Camera-space stacking works on debayered data. Because debayering is not stable with changes in illumination, debayered pixels are no longer completely linear so can produce artifacts. The output of this method is a linearized sRGB image with pixel values that can exceed 1.

    from pySP.image import RawDebayerDataFromRaw
    from pySP.raw_hdr import fuse_exposures_from_debayer
    from pySP.raw_bad_pixel_corr import find_erroneous_pixels_threshold, repair_bad_pixels, find_shared_pixels
    
    PATH_IMAGES_DNG = [...]
    
    images = [RawDebayerDataFromRaw(i) for i in PATH_IMAGES_DNG]
    image_hdr, image_count = fuse_exposures_from_debayer(images)

To render this image tonemapping is required. It is recommended that HDR images are exported and tonemapped in other software but a quick solution for debugging can be to use the Reinhardt tonemapper to produce an image that can be viewed correctly.

    from pySP.colorize import lin_srgb_to_srgb
    
    ...
    
    image_ldr =  image_hdr / (1  +  image_hdr)
    image_ldr_srgb =  lin_srgb_to_srgb(image_ldr)

## credits (and thanks)
 - Contributors of [rawpy](https://github.com/letmaik/rawpy) (and [libraw](https://www.libraw.org/)) for doing the hard work decoding raw images
 - [Nine Degrees Below](https://ninedegreesbelow.com/files/dcraw-c-code-annotated-code.html) for their excellent write-up on dcraw's raw processing
 - Adobe for making the [DNG specification](https://helpx.adobe.com/uk/camera-raw/digital-negative.html) both accessible and understandable

