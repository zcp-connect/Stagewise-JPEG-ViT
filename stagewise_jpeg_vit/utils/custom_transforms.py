import math
from typing import List, Tuple, Optional, Dict

import torch
from torch import Tensor
import utils.dct_ops as dops
import dct_manip as dm
import PIL.Image
from torchvision.transforms import functional as F, InterpolationMode

import logging, traceback

import itertools

###############################
#      RGB augmentations      #
#    DCT aug. are below RGB   #
###############################

def get_dimensions(img):
    width, height = F.get_image_size(img)
    channels = F.get_image_num_channels(img)
    return channels, height, width

def cutout(img, pad_size, replace=0):
    """Apply cutout (https://arxiv.org/abs/1708.04552) to image. 
    
    ### (PyTorch implementation of Google's big_vision cutout) ###
    
    This operation applies a (2*pad_size x 2*pad_size) mask of zeros to
    a random location within `img`. The pixel values filled in will be of the
    value `replace`. The located where the mask will be applied is randomly
    chosen uniformly over the whole image.
    Args:
        image: A PIL image
        pad_size: Specifies how big the zero mask that will be generated is that
        is applied to the image. The mask will be of size
        (2*pad_size x 2*pad_size).
        replace: What pixel value to fill in the image in the area that has
        the cutout mask applied to it.
    Returns:
        A PIL image of type uint8.
    """
    convert_back=False
    if F._is_pil_image(img):
        img = F.pil_to_tensor(img) # convert to tensor for pytorch operations
        convert_back=True
    assert img.dtype == torch.uint8, "PIL to tensor image is expected to have torch.unit8 as dtype."
    channels, height, width = get_dimensions(img)
    cutout_center_height = torch.randint(low=0, high=height, size=(1,)).item()
    cutout_center_width = torch.randint(low=0, high=width, size=(1,)).item()

    lower_pad = max(0, cutout_center_height - pad_size)
    upper_pad = max(0, height - cutout_center_height - pad_size)
    left_pad = max(0, cutout_center_width - pad_size)
    right_pad = max(0, width - cutout_center_width - pad_size)

    cutout_shape = (height - (lower_pad + upper_pad),
                    width - (left_pad + right_pad)) # cutout this shape
    padding_dims = (left_pad, right_pad, upper_pad, lower_pad)
    cutout_mask = torch.nn.functional.pad(
        torch.zeros(cutout_shape, dtype=img.dtype, device=img.device),
        padding_dims, value=1
    )
    cutout_mask = cutout_mask.unsqueeze(dim=0)
    cutout_mask = torch.tile(cutout_mask, (channels,1,1))
    img = torch.where(
        cutout_mask==0, # condition.
        torch.ones_like(img, dtype=img.dtype, device=img.device) * replace, # If true
        img # If condition is false
    )
    if convert_back:
        return F.to_pil_image(img)
    else:
        return img

def solarize_add(img, addition=0, threshold=128):
    """
    For each pixel in the image less than threshold
    we add 'addition' amount to it and then clip the
    pixel value to be between 0 and 255. The value
    of 'addition' is between -128 and 128.
    
    ### Re-implementation of Google's big_vision in PyTorch ###
    """
    convert_back=False
    if F._is_pil_image(img):
        img = F.pil_to_tensor(img) # convert to tensor for pytorch operations
        convert_back=True
    assert img.dtype == torch.uint8, "PIL to tensor image is expected to have torch.unit8 as dtype."
    added_img = img.to(torch.int) + addition
    added_img = torch.clamp(added_img, min=0,max=255)
    added_img = added_img.to(img.dtype)
    img = torch.where(
        img < threshold, # condition
        added_img, # if true
        img # if false
    )
    if convert_back:
        return F.to_pil_image(img)
    else:
        return img

def chroma_drop(img):
    img = img.convert("YCbCr")
    Y, Cb, Cr = img.split()
    if torch.rand(1).item() > 0.5:
        Cr = Cr.point(lambda i: 128)
    else:
        Cb = Cb.point(lambda i: 128)
    img = PIL.Image.merge("YCbCr", (Y, Cb, Cr))
    return img.convert("RGB")

def auto_saturation(img):
    img = img.convert("YCbCr")
    Y, Cb, Cr = img.split()
    Cbmin, Cbmax = Cb.getextrema()
    Crmin, Crmax = Cr.getextrema()
    Cmin = min(Cbmin, Crmin)
    Cmax = max(Cbmax, Crmax)
    Cb = Cb.point(lambda i: (i-Cmin) / (Cmax - Cmin) * 255 if (Cmax - Cmin) != 0 else i)
    Cr = Cr.point(lambda i: (i-Cmin) / (Cmax - Cmin) * 255 if (Cmax - Cmin) != 0 else i)
    img = PIL.Image.merge("YCbCr", (Y, Cb, Cr))
    return img.convert("RGB")

def _apply_op(
    img: Tensor, op_name: str, magnitude: float, interpolation: InterpolationMode, fill: Optional[List[float]]
):
    if op_name == "ShearX":
        # magnitude should be arctan(magnitude)
        # official autoaug: (1, level, 0, 0, 1, 0)
        # https://github.com/tensorflow/models/blob/dd02069717128186b88afa8d857ce57d17957f03/research/autoaugment/augmentation_transforms.py#L290
        # compared to
        # torchvision:      (1, tan(level), 0, 0, 1, 0)
        # https://github.com/pytorch/vision/blob/0c2373d0bba3499e95776e7936e207d8a1676e65/torchvision/transforms/functional.py#L976
        img = F.affine(
            img,
            angle=0.0,
            translate=[0, 0],
            scale=1.0,
            shear=[math.degrees(math.atan(magnitude)), 0.0],
            interpolation=interpolation,
            fill=fill,
            center=[0, 0],
        )
    elif op_name == "ShearY":
        # magnitude should be arctan(magnitude)
        # See above
        img = F.affine(
            img,
            angle=0.0,
            translate=[0, 0],
            scale=1.0,
            shear=[0.0, math.degrees(math.atan(magnitude))],
            interpolation=interpolation,
            fill=fill,
            center=[0, 0],
        )
    elif op_name == "TranslateX":
        img = F.affine(
            img,
            angle=0.0,
            translate=[int(magnitude), 0],
            scale=1.0,
            interpolation=interpolation,
            shear=[0.0, 0.0],
            fill=fill,
        )
    elif op_name == "TranslateY":
        img = F.affine(
            img,
            angle=0.0,
            translate=[0, int(magnitude)],
            scale=1.0,
            interpolation=interpolation,
            shear=[0.0, 0.0],
            fill=fill,
        )
    elif op_name == "Rotate":
        img = F.rotate(img, magnitude, interpolation=interpolation, fill=fill)
    elif op_name == "Brightness":
        img = F.adjust_brightness(img, 1.0 + magnitude)
    elif op_name == "Color":
        img = F.adjust_saturation(img, 1.0 + magnitude)
    elif op_name == "Contrast":
        img = F.adjust_contrast(img, 1.0 + magnitude)
    elif op_name == "Sharpness":
        img = F.adjust_sharpness(img, 1.0 + magnitude)
    elif op_name == "Posterize":
        img = F.posterize(img, int(magnitude))
    elif op_name == "Solarize":
        img = F.solarize(img, magnitude)
    elif op_name == "AutoContrast":
        img = F.autocontrast(img)
    elif op_name == "Equalize":
        img = F.equalize(img)
    elif op_name == "Invert":
        img = F.invert(img)
    elif op_name == "Identity":
        pass
    elif op_name == 'Cutout': # added
        img = cutout(img, int(magnitude), replace=fill)
    elif op_name == "SolarizeAdd": # added
        img = solarize_add(img, int(magnitude))
    elif op_name == "Grayscale": # added v2
        img = F.to_grayscale(img, num_output_channels=3)
    elif op_name == "ChromaDrop": #
        img = chroma_drop(img)
    elif op_name == "AutoSaturation":
        #img = auto_saturation(img)
        img = auto_saturation(img) # dct-equivalent
    elif op_name == "AutoSaturation_old": # for compatibility purposes
        img = auto_saturation(img)
    elif op_name == "Rotate90": # magnitude is +- 90
        img = F.rotate(img, magnitude, interpolation=interpolation, fill=fill)
    else:
        raise ValueError(f"The provided operator {op_name} is not recognized.")
    return img



class RandAugment_bv(torch.nn.Module):
    r"""RandAugment data augmentation method based on
    `"RandAugment: Practical automated data augmentation with a reduced search space"
    <https://arxiv.org/abs/1909.13719>`_.

    ### Re-implementation of Google's Big Vision randaugment in PyTorch ###

    If the image is torch Tensor, it should be of type torch.uint8, and it is expected
    to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.
    If img is PIL Image, it is expected to be in mode "L" or "RGB".

    Args:
        num_ops (int): Number of augmentation transformations to apply sequentially.
        magnitude (int): Magnitude for all the transformations.
        num_magnitude_bins (int): The number of different magnitude values.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.NEAREST``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` are supported.
        fill (sequence or number, optional): Pixel fill value for the area outside the transformed
            image. If given a number, the value is used for all bands respectively.
    """

    def __init__(
        self,
        num_ops: int = 2,
        magnitude: int = 10,
        num_magnitude_bins: int = 11,
        interpolation: InterpolationMode = InterpolationMode.NEAREST,
        fill: Optional[List[float]] = None,
        ops_list = ["AutoContrast", "Equalize", "Invert", "Rotate", "Posterize", "Solarize", "SolarizeAdd", "Color", "Contrast", "Brightness",
                        "Sharpness", "ShearX", "ShearY", "Cutout", "TranslateX", "TranslateY"]
    ) -> None:
        super().__init__()
        self.num_ops = num_ops
        self.magnitude = magnitude
        self.num_magnitude_bins = num_magnitude_bins
        self.interpolation = interpolation
        self.fill = fill
        if ops_list==None:
            self.ops_list = ["AutoContrast", "Equalize", "Invert", "Rotate", "Posterize", "Solarize", "SolarizeAdd", "Color", "Contrast", "Brightness",
                        "Sharpness", "ShearX", "ShearY", "Cutout", "TranslateX", "TranslateY"]
        else:
            self.ops_list = ops_list

    def _augmentation_space(self, num_bins: int, image_size: Tuple[int, int]) -> Dict[str, Tuple[Tensor, bool]]:
        return {
            # op_name: (magnitudes, signed)
            #"Identity": (torch.tensor(0.0), False), not needed
            "AutoContrast": (torch.tensor(0.0), False),
            "Equalize": (torch.tensor(0.0), False),
            "Invert": (torch.tensor(0.0), False), # added
            "Rotate": (torch.linspace(0.0, 30.0, num_bins), True),
            "Posterize": (8 - (torch.arange(num_bins) / ((num_bins - 1) / 4)).round().int(), False),
            "Solarize": (torch.linspace(255.0, 0.0, num_bins), False),
            "SolarizeAdd": (torch.linspace(0, 110, num_bins), False), # added
            "Color": (torch.linspace(0.0, 0.9, num_bins), True),
            "Contrast": (torch.linspace(0.0, 0.9, num_bins), True),
            "Brightness": (torch.linspace(0.0, 0.9, num_bins), True),
            "Sharpness": (torch.linspace(0.0, 0.9, num_bins), True),
            "ShearX": (torch.linspace(0.0, 0.3, num_bins), True),
            "ShearY": (torch.linspace(0.0, 0.3, num_bins), True),
            "Cutout": (torch.linspace(0, 40, num_bins), False), #added
            "TranslateX": (torch.linspace(0.0, 150.0 / 336.0 * image_size[1], num_bins), True),
            "TranslateY": (torch.linspace(0.0, 150.0 / 336.0 * image_size[0], num_bins), True),
            "Grayscale": (torch.tensor(0.0), False),
            "ChromaDrop": (torch.tensor(0.0), False),
            "AutoSaturation": (torch.tensor(0.0), False),
            "AutoSaturation_old": (torch.tensor(0.0), False),
            "Rotate90": (torch.tensor(90.0), True),
        }


    def forward(self, img: Tensor) -> Tensor:
        """
            img (PIL Image or Tensor): Image to be transformed.

        Returns:
            PIL Image or Tensor: Transformed image.
        """
        fill = self.fill
        channels, height, width = get_dimensions(img)

        op_meta = self._augmentation_space(self.num_magnitude_bins, (height, width))
        for _ in range(self.num_ops):
            op_index = int(torch.randint(len(self.ops_list), (1,)).item())
            op_name = list(self.ops_list)[op_index]
            magnitudes, signed = op_meta[op_name]
            magnitude = float(magnitudes[self.magnitude].item()) if magnitudes.ndim > 0 else 0.0
            if signed and torch.randint(2, (1,)):
                magnitude *= -1.0
            img = _apply_op(img, op_name, magnitude, interpolation=self.interpolation, fill=fill)

        return img


    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"num_ops={self.num_ops}"
            f", magnitude={self.magnitude}"
            f", num_magnitude_bins={self.num_magnitude_bins}"
            f", interpolation={self.interpolation}"
            f", fill={self.fill}"
            f")"
        )
        return s


class ToTensor_range(torch.nn.Module):
    r"""
    Converts PIL image to Tensor into a specified range

    Args:
        val_min = minimum value after convert
        val_max = maximum value after convert
        dtype = dtype after convert (default=torch.float32)

    Returns:
        Converted Torch Tensor
    """

    def __init__(
        self,
        val_min: float = -1.,
        val_max: float = 1.,
        dtype = torch.float32,
    ) -> Tensor:
        super().__init__()
        self.val_min = val_min
        self.val_max = val_max
        self.dtype = dtype

    def forward(self, img) -> Tensor:
        """
            img (PIL Image): Image to be transformed.

        Returns:
            Tensor: Converted Image
        """
        if F._is_pil_image(img):
            img = F.to_tensor(img) # to_tensor normalizes data to (0,1)
        img = img.to(self.dtype) # convert dtype
        img = self.val_min + (img * (self.val_max - self.val_min)) # scale to val_min to val_max

        return img

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"val_min={self.val_min}"
            f", val_max={self.val_max}"
            f", dtype={self.dtype}"
            f")"
        )
        return s

###############################
#      DCT augmentations      #
###############################

def dct_prep(coeff:Tensor, mode='start'):
    """
    preps dct coeff for augmentation

    if tuple -> list
    if element -> list (one item)

    returns in original data structure

    Args:
        coeff: DCT coefficient
        mode: 'start' or 'end'
    """
    if mode=='start':
        if type(coeff) == tuple: # if tuple, convert to list for in-place op
            return list(coeff)
        else: # if not, create single item list
            return [coeff]
    elif mode=='end':
        if len(coeff) > 1: # if tuple->list, convert it back to tuple
            return tuple(coeff)
        else: # if single item, return single item without list wrapper
            return coeff[0]

class ToRange(torch.nn.Module):
    """
    Scales the value of a Tensor into a specified range / converts dtype
    
    Args:
        val_min = minimum value after convert
        val_max = maximum value after convert
        orig_min = minimum value before convert
        orig_max = maximum value before convert
        dtype = dtype after convert (default=torch.float32)

    Returns:
        Converted Torch Tensor
    """

    def __init__(
        self,
        val_min: float = -1.,
        val_max: float = 1.,
        orig_min: float = -1024,
        orig_max: float = 1024,
        dtype = torch.float32,
    ) -> Tensor:
        super().__init__()
        self.val_min = val_min
        self.val_max = val_max
        self.orig_min = orig_min
        self.orig_max = orig_max
        self.dtype = dtype

    def forward(self, img) -> Tensor:
        """
        Args:
            img (tuple or Tensor): Tensor to be scaled

        Returns:
            Tensor: Scaled Tensor
        """
        img = dct_prep(img, mode='start') # convert it to list
            
        for i in range(len(img)):
            try:
                img[i] = img[i].to(self.dtype) # convert dtype
            except:
                print(f"Error! {type(img)} {type(img[i])} {len(img)} {len(img[i])} {i}")
            img[i] = (img[i] - self.orig_min) / (self.orig_max - self.orig_min) # scale from 0 to 1 (16-bit for DCT after dequantization -- but still in range -1024~1016)
            img[i] = self.val_min + (img[i] * (self.val_max - self.val_min)) # scale to val_min to val_max
        
        return dct_prep(img, mode='end') # convert it back to original data structure

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"val_min={self.val_min}"
            f", val_max={self.val_max}"
            f", orig_min={self.orig_min}"
            f", orig_max={self.orig_max}"
            f", dtype={self.dtype}"
            f")"
        )
        return s

class Resize_DCT(torch.nn.Module):
    """
    Scales the size of dct coefficients to a specified size
    
    Args:
        size: desired size of the dct coefficients after transformation
        chroma_scale: scale down 'size' by this amount
        strict_even_size: If true, forces the 'size' to be even

    Returns:
        Resized dct coefficient
    """
    def __init__(
        self,
        size,
        chroma_scale=2,
        dtype_resize=torch.float32,
        strict_even_size=False,
    ) -> Tensor:
        super().__init__()
        self.size = size
        self.chroma_scale = chroma_scale
        self.strict_even_size=strict_even_size
        self.dtype_resize=dtype_resize
        self.conv_mxs = dict()
        if self.strict_even_size:
            assert self.size % 2 == 0, f"ERROR: Resize_dct should have even numbered 'size' parameter. Current size: {size}"

    def forward(self, coeff) -> Tensor:
        """
        Args:
            coeff (tuple or Tensor): Tensor containing DCT coefficients to be scaled (must have five dimensions: (c, h, w, kh, kw))

        Returns:
            Tensor: Scaled Tensor
        """

        coeff = dct_prep(coeff, mode='start') # convert it to list

        for i in range(len(coeff)):
            assert len(coeff[i].shape) == 5, "DCT coefficients should have 5 dimensions of (C, H, W, KH, KW) where KH, KW are typically 8"
            c, _, _, _, _ = coeff[i].shape
            size_to_resize = self.size
            if c==2: # if it is a chroma channel with c=2
                size_to_resize = math.ceil(size_to_resize / self.chroma_scale)
            coeff[i] = dops.resize_dct(coeff[i], size_to_resize, dtype=self.dtype_resize, conv_mxs=self.conv_mxs)
        
        return dct_prep(coeff, mode='end') # return original format

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"size={self.size}"
            f", smaller_chroma={self.smaller_chroma}"
            f", strict_even_size={self.strict_even_size}"
            f")"
        )
        return s

class RandomResizedCrop_DCT(torch.nn.Module):
    """
    Crop DCT coefficients at a random location

    expected shape = 5 dims of (c, h, w, kh, kw)

    Args:
        size: size after resize
        scale: lower & upper bounds for random area of the crop before resizing
        ratio: lower & upper bounds for the aspect ratio of the crop before resizing
        dtype_resize: dtype to use during resizing
        chroma_scale: scale down random crop box for chroma channel
    """
    def __init__(
        self, size, scale=(0.05, 1.0), ratio=(3.0/4.0, 4.0/3.0), dtype_resize=torch.float32, chroma_scale=2):
        super().__init__()
        self.size = size
        self.scale = scale
        self.ratio = ratio
        self.dtype_resize = dtype_resize
        self.chroma_scale = chroma_scale
        self.conv_mxs = dict()

        flatten_iter = itertools.chain.from_iterable # get factors for 'size'
        def factors(n):
            return list(flatten_iter((i, n//i) 
                        for i in range(1, int(n**0.5)+1) if n % i == 0))
        self.size_choices, _ = torch.tensor(factors(size)).sort()
        self.even_size_choices, _ = torch.tensor([c for c in self.size_choices if c%2==0]).sort()

    @staticmethod
    def get_params(coeff, scale: List[float], ratio: List[float], choices: list, chroma_scale=2) -> Tuple[int, int, int, int]:
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            coeff (Tensor): dct coefficient with five channels in (c, h, w, kh, kw)
            scale (list): range of scale of the origin size cropped
            ratio (list): range of aspect ratio of the origin aspect ratio cropped
            chroma_scale (int): scale chroma coordinates to the inverse of this parameter (e.g. if 2 = i,j,h,w /= 2)

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
            sized crop.
        """
        def choose_closest(val: int, choices: list, maxval: int):
            if val <= choices[-1]: # if val <= target size
                closest = choices[torch.argmin(torch.abs(choices-val))] # choose closest factor of size
            else: # if val > target_size
                closest = torch.round(val/choices[-1]).item()*choices[-1] # choose closest multiple of size
                if closest > maxval:
                    closest -= choices[-1]
            return closest

        c, height, width, _, _ = coeff.shape
        area = height * width

        skip_ratio=False # flag to avoid expensive exp and log calls if ratio is fixed
        if ratio[0] == ratio[1] and ratio[1] == 1:
            skip_ratio=True
        if not skip_ratio:
            log_ratio = torch.log(torch.tensor(ratio))

        for _ in range(10):
            target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            if not skip_ratio:
                aspect_ratio = torch.exp(torch.empty(1).uniform_(log_ratio[0], log_ratio[1])).item()
                w = int(round(math.sqrt(target_area * aspect_ratio)))
                h = int(round(math.sqrt(target_area / aspect_ratio)))
                w = choose_closest(w, choices, width)
                h = choose_closest(h, choices, height)
            else: # aspect ratio is 1 if this flag is true
                w = int(round(math.sqrt(target_area)))
                w = choose_closest(w, choices, width)
                h = w

            w = int(max(2, w)) # minimum size = 2 (to avoid zero-area cropping bug)
            h = int(max(2, h))

            if w <= width and h <= height:
                i = int(torch.randint(0, height - h + 1, size=(1,)).item() // chroma_scale * chroma_scale)
                j = int(torch.randint(0, width - w + 1, size=(1,)).item() // chroma_scale * chroma_scale)
                w = int(max(1, w)) # absolute minimum size to crop = 1
                h = int(max(1, h))
                return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        h = choose_closest(h, choices, height)
        w = choose_closest(w, choices, width)
        i = int(torch.div((height - h), 2, rounding_mode='floor').div(chroma_scale, rounding_mode='floor') * chroma_scale)
        j = int(torch.div((width - w), 2, rounding_mode='floor').div(chroma_scale, rounding_mode='floor') * chroma_scale)
        h = int(max(1, h)) # min area to crop = 1
        w = int(max(1, w)) # min area to crop = 1
        return i, j, h, w

    def forward(self, coeff):
        """
        Args:
            coeff (Tensor or tuple): dct coefficient with five channels in (c, h, w, kh, kw)
                                    if tuple, (Y, cbcr) is expected.

        Returns:
            Tensor: Randomly cropped and resized image.
        """
        coeff = dct_prep(coeff, mode='start') # convert it to list

        for idx in range(len(coeff)):
            c, _, _, _, _ = coeff[idx].shape
            size_to_resize = self.size
            if c==1 or len(coeff) == 1: # Get boxcoord if c==1 (should be the first element in tuple) or only one element
                i, j, h, w = self.get_params(coeff[idx], self.scale, self.ratio, self.even_size_choices, self.chroma_scale)
            if c==2: # if it is a chroma channel with c=2
                size_to_resize = math.ceil(size_to_resize / self.chroma_scale)
                i //= self.chroma_scale
                j //= self.chroma_scale
                h //= self.chroma_scale
                w //= self.chroma_scale
            h = max(1, h) # min area to crop = 1
            w = max(1, w) # min area to crop = 1
            coeff[idx] = dops.crop_dct(coeff[idx], i, j, h, w)
            #print(i, j, h, w)
            try:
                coeff[idx] = dops.resize_dct(coeff[idx], size_to_resize, dtype=self.dtype_resize, conv_mxs=self.conv_mxs)
            except (BaseException) as e:
                logging.error(traceback.format_exc())
                print(f"\n Erroneous Data: {coeff[idx]} \n\n {coeff[idx].shape} \n\n Idx: {idx} \n\n i j h w: {i} {j} {h} {w}")
        
        return dct_prep(coeff, mode='end') # return original format

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + f"(size={self.size}"
        format_string += f", scale={tuple(round(s, 4) for s in self.scale)}"
        format_string += f", ratio={tuple(round(r, 4) for r in self.ratio)}"
        return format_string

class RandomCrop_DCT(torch.nn.Module):
    """
    Crop DCT coefficients at a random location

    expected shape = 5 dims of (c, h, w, kh, kw)

    Args:
        size: size of cropping window (height and width)
        chroma_scale: scale down random crop box for chroma channel
    """
    def __init__(
        self, size, chroma_scale=2):
        super().__init__()
        self.size = size
        self.chroma_scale = chroma_scale

    @staticmethod
    def get_params(coeff, size, chroma_scale=2) -> Tuple[int, int, int, int]:
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            coeff (Tensor): dct coefficient with five channels in (c, h, w, kh, kw)
            size: size of window to crop
            chroma_scale (int): scale chroma coordinates to the inverse of this parameter (e.g. if 2 = i,j,h,w /= 2)

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
            sized crop.
        """
        c, height, width, _, _ = coeff.shape

        h, w = size, size
        assert w <= width and h <= height, f"Crop window should be smaller than original image's height and width. Current window size: {h},{w}, Original image size: {height},{width}"
        i = int(torch.randint(0, height - h + 1, size=(1,)).item())
        j = int(torch.randint(0, width - w + 1, size=(1,)).item())
        if c==1: # if it is luma channel and needs to be forced to a certain chroma scale
            i = i // chroma_scale * chroma_scale
            j = j // chroma_scale * chroma_scale
        w = int(max(1, w)) # absolute minimum size to crop = 1
        h = int(max(1, h))
        return i, j, h, w
        

    def forward(self, coeff):
        """
        Args:
            coeff (Tensor or tuple): dct coefficient with five channels in (c, h, w, kh, kw)
                                    if tuple, (Y, cbcr) is expected.

        Returns:
            Tensor: Randomly cropped and resized image.
        """
        coeff = dct_prep(coeff, mode='start') # convert it to list

        for idx in range(len(coeff)):
            c, _, _, _, _ = coeff[idx].shape
            size_to_resize = self.size
            if c==1 or len(coeff) == 1: # Get boxcoord if c==1 (should be the first element in tuple) or only one element
                i, j, h, w = self.get_params(coeff[idx], self.size, self.chroma_scale)
            if c==2: # if it is a chroma channel with c=2
                size_to_resize = math.ceil(size_to_resize / self.chroma_scale)
                i //= self.chroma_scale
                j //= self.chroma_scale
                h //= self.chroma_scale
                w //= self.chroma_scale
            h = max(1, h) # min area to crop = 1
            w = max(1, w) # min area to crop = 1
            coeff[idx] = dops.crop_dct(coeff[idx], i, j, h, w)
        
        return dct_prep(coeff, mode='end') # return original format

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + f"(size={self.size}"
        format_string += f", chroma_scale={self.chroma_scale})"
        return format_string

class CenterCrop_DCT(torch.nn.Module):
    """
    Center DCT Crop

    Args:
        Size: Size of the center crop (should be multiple of chroma_scale, by default even.)
        chroma_scale: factor to scale down chroma cropping
        interpolation: interpolation mode
    """
    def __init__(
        self, size, chroma_scale=2
    ):
        super().__init__()
        self.size = size
        self.chroma_scale = chroma_scale
    
    @staticmethod
    def get_params(coeff, size, chroma_scale=2) -> Tuple[int, int, int, int]:
        """Get parameters for ``crop`` for a center crop.

        Args:
            coeff (Tensor): dct coefficient with five channels in (c, h, w, kh, kw)
            size (int): center crop size
            chroma_scale (int): scale chroma coordinates to the inverse of this parameter (e.g. if 2 = i,j,h,w /= 2)

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a center crop
        """
        c, height, width, _, _ = coeff.shape

        w = size
        h = size
        assert w <= width and h <= height, f"Crop window should be smaller than original image's height and width. Current window size: {h},{w}, Original image size: {height},{width}"
        i = int(height - size) // 2
        j = int(width - size) // 2
        if c==1:  # if it is luma channel and needs to be forced to a certain chroma scale
            i = i // chroma_scale * chroma_scale
            j = j // chroma_scale * chroma_scale
            h = h //chroma_scale * chroma_scale
            w = w //chroma_scale * chroma_scale
        h = max(1, h) # min area to crop = 1
        w = max(1, w) # min area to crop = 1
        return i, j, h, w
    
    def forward(self, coeff):
        """
        Args:
            coeff (Tensor or tuple): dct coefficient with five channels in (c, h, w, kh, kw)
                                    if tuple, (Y, cbcr) is expected.

        Returns:
            Tensor: Center cropped and resized image.
        """
        coeff = dct_prep(coeff, mode='start') # convert it to list

        for idx in range(len(coeff)):
            c, _, _, _, _ = coeff[idx].shape
            size_to_resize = self.size
            if c==1 or len(coeff) == 1: # Get boxcoord if c==1 (should be the first element in tuple) or only one element
                i, j, h, w = self.get_params(coeff[idx], self.size, self.chroma_scale)
            if c==2: # if it is a chroma channel with c=2
                size_to_resize = math.ceil(size_to_resize / self.chroma_scale)
                i //= self.chroma_scale
                j //= self.chroma_scale
                h //= self.chroma_scale
                w //= self.chroma_scale
            h = max(1, h) # min area to crop = 1
            w = max(1, w) # min area to crop = 1
            coeff[idx] = dops.crop_dct(coeff[idx], i, j, h, w)
        
        return dct_prep(coeff, mode='end') # return original format

class ResizedCenterCrop_DCT(torch.nn.Module):
    """
    Resized Center DCT Crop -- for more efficient resizing/cropping
    Process:
        1. Crop to (size_crop/size_resize) * H/W
        2. Resize to size_crop

    Args:
        size_resize: size to resize (pseudo-resize)
        size_crop: size to center-crop
        chroma_scale: factor to scale down chroma cropping
        interpolation: interpolation mode
    """
    def __init__(
        self, size_resize, size_crop, chroma_scale=2, dtype_resize=torch.float32,
    ):
        super().__init__()
        self.size_resize = size_resize
        self.size_crop = size_crop
        self.chroma_scale = chroma_scale
        self.dtype_resize = dtype_resize
        self.size_ratio = size_crop / size_resize
        self.conv_mxs = dict()

        flatten_iter = itertools.chain.from_iterable # get factors for 'size'
        def factors(n):
            return list(flatten_iter((i, n//i) 
                        for i in range(1, int(n**0.5)+1) if n % i == 0))
        self.size_choices, _ = torch.tensor(factors(size_crop)).sort()
        self.even_size_choices, _ = torch.tensor([c for c in self.size_choices if c%2==0]).sort()
    
    def get_params(self, coeff, chroma_scale=2) -> Tuple[int, int, int, int]:
        """Get parameters for ``crop`` for a center crop.

        Args:
            coeff (Tensor): dct coefficient with five channels in (c, h, w, kh, kw)
            chroma_scale (int): scale chroma coordinates to the inverse of this parameter (e.g. if 2 = i,j,h,w /= 2)

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a center crop
        """
        def choose_closest(val: int, choices: list, maxval: int):
            if val <= choices[-1]: # if val <= target size
                closest = choices[torch.argmin(torch.abs(choices-val))] # choose closest factor of size
            else: # if val > target_size
                closest = torch.round(val/choices[-1]).item()*choices[-1] # choose closest multiple of size
                if closest > maxval:
                    closest -= choices[-1]
            return closest

        c, height, width, _, _ = coeff.shape

        w = round(self.size_ratio * width)
        h = round(self.size_ratio * height)
        w = choose_closest(w, self.even_size_choices, width)
        h = choose_closest(h, self.even_size_choices, height)
        i = int(torch.div((height - h), 2, rounding_mode='floor'))
        j = int(torch.div((width - w), 2, rounding_mode='floor'))
        if c==1:  # if it is luma channel and needs to be forced to a certain chroma scale
            i = i // chroma_scale * chroma_scale
            j = j // chroma_scale * chroma_scale
        h = int(max(1, h)) # min area to crop = 1
        w = int(max(1, w)) # min area to crop = 1
        return i, j, h, w
    
    def forward(self, coeff):
        """
        Args:
            coeff (Tensor or tuple): dct coefficient with five channels in (c, h, w, kh, kw)
                                    if tuple, (Y, cbcr) is expected.

        Returns:
            Tensor: Center cropped and resized image.
        """
        coeff = dct_prep(coeff, mode='start') # convert it to list

        for idx in range(len(coeff)):
            c, _, _, _, _ = coeff[idx].shape
            size_to_resize = self.size_crop
            if c==1 or len(coeff) == 1: # Get boxcoord if c==1 (should be the first element in tuple) or only one element
                i, j, h, w = self.get_params(coeff[idx], self.chroma_scale)
            if c==2: # if it is a chroma channel with c=2
                size_to_resize = math.ceil(size_to_resize / self.chroma_scale)
                i //= self.chroma_scale
                j //= self.chroma_scale
                h //= self.chroma_scale
                w //= self.chroma_scale
            h = max(1, h) # min area to crop = 1
            w = max(1, w) # min area to crop = 1
            coeff[idx] = dops.crop_dct(coeff[idx], i, j, h, w)
            coeff[idx] = dops.resize_dct(coeff[idx], size_to_resize, dtype=self.dtype_resize, conv_mxs=self.conv_mxs)
        
        return dct_prep(coeff, mode='end') # return original format

class RandomFlip_DCT(torch.nn.Module):
    """
    Random DCT Flip

    Args:
        p: probability to apply flipping operation
        direction: 'horizontal' or 'vertical'. Direction of the flip
    """
    def __init__(self, p=0.5, direction='horizontal'):
        super().__init__()
        self.p = p
        self.direction = direction

    def forward(self, coeff):
        """
        Args:
            coeff (Tensor or tuple): dct coefficient with five channels in (c, h, w, kh, kw)
                                    if tuple, (Y, cbcr) is expected.
        Returns:
            Tensor: Flipped DCT coefficient
        """
        if torch.rand(1) > self.p: # (1-p): don't apply p: apply
            return coeff

        coeff = dct_prep(coeff, mode='start') # convert it to list
        
        for i in range(len(coeff)):
            coeff[i] = dops.flip_dct(coeff[i], direction=self.direction) # flip each dct channels (Y, cbcr) or just one
        
        return dct_prep(coeff, mode='end') # return original format

class DCDiffQuantDCT(torch.nn.Module):
    """
    Replace each quantized DC coefficient with a component-wise raster DPCM value.

    This transform is intended to be inserted after geometric DCT transforms and
    before coefficient-space RandAugment. Later AC-only augmentations can then
    change AC values without changing how DC differences are computed.
    """
    def __init__(self, mode: str = "standard"):
        super().__init__()
        if mode != "standard":
            raise ValueError(f"Unsupported DCDiffQuantDCT mode: {mode}")
        self.mode = mode

    @staticmethod
    def _component_raster_diff(values: Tensor) -> Tensor:
        diffs = torch.empty_like(values)
        diffs[0] = values[0]
        diffs[1:] = values[1:] - values[:-1]
        return diffs

    def forward(self, coeff: tuple) -> Tensor:
        coeff = dct_prep(coeff, mode='start')
        y_component = coeff[0].clone()
        _, y_height, y_width, _, _ = y_component.shape
        even_height = y_height - (y_height % 2)
        even_width = y_width - (y_width % 2)
        y_dc = y_component[:, :even_height, :even_width, 0, 0]
        y_sequence = y_dc.reshape(1, even_height // 2, 2, even_width // 2, 2).permute(0, 1, 3, 2, 4).reshape(-1)
        y_diff = self._component_raster_diff(y_sequence).reshape(1, even_height // 2, even_width // 2, 2, 2)
        y_component[:, :even_height, :even_width, 0, 0] = y_diff.permute(0, 1, 3, 2, 4).reshape(1, even_height, even_width)
        coeff[0] = y_component

        if len(coeff) > 1:
            cbcr_component = coeff[1].clone()
            for component_index in range(cbcr_component.shape[0]):
                dc_values = cbcr_component[component_index, :, :, 0, 0].reshape(-1)
                cbcr_component[component_index, :, :, 0, 0] = self._component_raster_diff(dc_values).reshape(cbcr_component.shape[1:3])
            coeff[1] = cbcr_component

        return dct_prep(coeff, mode='end')

class RandAugment_QuantDCT(torch.nn.Module):
    """
    Quantized-DCT-safe augmentation.

    This class intentionally avoids DCT-basis mixing operations such as resize,
    rotate, shear, histogram equalization, and solarize. It operates directly on
    JPEG quantized DCT coefficient integers.
    """

    def __init__(
        self,
        num_ops: int = 2,
        magnitude: int = 3,
        num_magnitude_bins: int = 11,
        ops_list = None,
        max_abs: int = 189,
    ) -> None:
        super().__init__()
        self.num_ops = num_ops
        self.magnitude = magnitude
        self.num_magnitude_bins = num_magnitude_bins
        self.max_abs = max_abs
        if ops_list is None:
            self.ops_list = [
                "TranslateX",
                "TranslateY",
                "Cutout",
                "Grayscale",
                "ChromaDrop",
                "BlockDropout",
                "FreqBandDropout",
                "ACQuantNoise",
                "FreqEnhance",
            ]
        else:
            self.ops_list = ops_list

    def _augmentation_space(self, num_bins: int, image_size: Tuple[int, int]) -> Dict[str, Tuple[Tensor, bool]]:
        max_translate = max(2, min(6, min(image_size) // 4))
        return {
            "Identity": (torch.tensor(0.0), False),
            "TranslateX": (torch.linspace(0.0, float(max_translate), num_bins), True),
            "TranslateY": (torch.linspace(0.0, float(max_translate), num_bins), True),
            "Cutout": (torch.linspace(0.0, 6.0, num_bins), False),
            "Grayscale": (torch.tensor(0.0), False),
            "ChromaDrop": (torch.tensor(0.0), False),
            "BlockDropout": (torch.linspace(0.0, 0.15, num_bins), False),
            "FreqBandDropout": (torch.linspace(14.0, 6.0, num_bins), False),
            "ACQuantNoise": (torch.linspace(0.0, 0.15, num_bins), False),
            "Brightness": (torch.linspace(0.0, 0.4, num_bins), True),
            "Contrast": (torch.linspace(0.0, 0.4, num_bins), True),
            "Color": (torch.linspace(0.0, 0.4, num_bins), True),
            "Sharpness": (torch.linspace(0.0, 0.5, num_bins), True),
            "MidfreqAug": (torch.linspace(0.0, 0.5, num_bins), True),
            "FreqEnhance": (torch.linspace(0.0, 0.5, num_bins), True),
        }

    def _finalize_component(self, component: Tensor, dtype: torch.dtype) -> Tensor:
        component = component.round().clamp(min=-self.max_abs, max=self.max_abs)
        return component.to(dtype=dtype).contiguous()

    def _finalize(self, coeff, dtypes):
        for component_index in range(len(coeff)):
            coeff[component_index] = self._finalize_component(coeff[component_index], dtypes[component_index])
        return coeff

    def _translate(self, coeff, magnitude, direction):
        transblocks = int(magnitude - (magnitude % 2))
        if transblocks == 0:
            return coeff
        coeff[0] = dops.translate_dct(coeff[0], transblocks, direction=direction)
        if len(coeff) > 1:
            coeff[1] = dops.translate_dct(coeff[1], transblocks // 2, direction=direction)
        return coeff

    def _cutout(self, coeff, magnitude):
        cutout_size = int(round(magnitude))
        cutout_size = int(cutout_size - (cutout_size % 2))
        if cutout_size == 0:
            return coeff
        coeff[0], center_height, center_width = dops.cutout_dct(coeff[0], cutout_size, replace=0)
        if len(coeff) > 1:
            coeff[1], _, _ = dops.cutout_dct(coeff[1], cutout_size // 2, 0, center_height // 2, center_width // 2)
        return coeff

    def _block_dropout(self, coeff, probability):
        if probability <= 0:
            return coeff
        if len(coeff) > 1:
            _, chroma_height, chroma_width, _, _ = coeff[1].shape
            mcu_mask = torch.rand((chroma_height, chroma_width), device=coeff[1].device) < probability
            y_mask = torch.repeat_interleave(torch.repeat_interleave(mcu_mask, 2, dim=0), 2, dim=1)
            y_mask = y_mask[:coeff[0].shape[1], :coeff[0].shape[2]]
            coeff[0] = torch.where(y_mask.unsqueeze(0).unsqueeze(-1).unsqueeze(-1), torch.zeros_like(coeff[0]), coeff[0])
            coeff[1] = torch.where(mcu_mask.unsqueeze(0).unsqueeze(-1).unsqueeze(-1), torch.zeros_like(coeff[1]), coeff[1])
            return coeff
        _, height, width, _, _ = coeff[0].shape
        block_mask = torch.rand((height, width), device=coeff[0].device) < probability
        coeff[0] = torch.where(block_mask.unsqueeze(0).unsqueeze(-1).unsqueeze(-1), torch.zeros_like(coeff[0]), coeff[0])
        return coeff

    def _freq_band_dropout(self, coeff, threshold):
        for component_index in range(len(coeff)):
            _, _, _, block_height, block_width = coeff[component_index].shape
            freq_rows = torch.arange(block_height, device=coeff[component_index].device).view(block_height, 1)
            freq_cols = torch.arange(block_width, device=coeff[component_index].device).view(1, block_width)
            freq_mask = (freq_rows + freq_cols) >= int(round(threshold))
            freq_mask[0, 0] = False
            coeff[component_index] = torch.where(
                freq_mask.view(1, 1, 1, block_height, block_width),
                torch.zeros_like(coeff[component_index]),
                coeff[component_index],
            )
        return coeff

    def _ac_quant_noise(self, coeff, probability):
        if probability <= 0:
            return coeff
        for component_index in range(len(coeff)):
            component = coeff[component_index]
            _, _, _, block_height, block_width = component.shape
            ac_mask = torch.ones((block_height, block_width), dtype=torch.bool, device=component.device)
            ac_mask[0, 0] = False
            random_mask = torch.rand(component.shape, device=component.device) < probability
            nonzero_mask = component != 0
            noise = torch.randint(-1, 2, component.shape, device=component.device, dtype=component.dtype)
            update_mask = random_mask & nonzero_mask & ac_mask.view(1, 1, 1, block_height, block_width)
            coeff[component_index] = component + torch.where(update_mask, noise, torch.zeros_like(component))
        return coeff

    def _scale_dc(self, component, factor):
        component = component.clone()
        component[:, :, :, 0, 0] = component[:, :, :, 0, 0] * factor
        return component

    def _brightness(self, coeff, magnitude):
        y_component = coeff[0].clone()
        dc_values = y_component[:, :, :, 0, 0]
        delta = torch.round(dc_values.abs().float().mean().clamp_min(1.0) * magnitude * 0.25)
        y_component[:, :, :, 0, 0] = dc_values + delta.to(dtype=dc_values.dtype)
        coeff[0] = y_component
        return coeff

    def _scale_freq(self, coeff, magnitude, mode):
        factor = max(0.0, 1.0 + magnitude)
        for component_index in range(len(coeff)):
            component = coeff[component_index].clone()
            _, _, _, block_height, block_width = component.shape
            freq_rows = torch.arange(block_height, device=component.device, dtype=component.dtype).view(block_height, 1)
            freq_cols = torch.arange(block_width, device=component.device, dtype=component.dtype).view(1, block_width)
            freq_sum = freq_rows + freq_cols
            if mode == "ac":
                freq_mask = torch.ones((block_height, block_width), dtype=torch.bool, device=component.device)
                freq_mask[0, 0] = False
                component = torch.where(
                    freq_mask.view(1, 1, 1, block_height, block_width),
                    component * factor,
                    component,
                )
            elif mode == "mid":
                freq_mask = (freq_sum >= 3) & (freq_sum <= 8)
                component = torch.where(
                    freq_mask.view(1, 1, 1, block_height, block_width),
                    component * factor,
                    component,
                )
            elif mode == "sharp":
                normalizer = max(1.0, float(block_height + block_width - 2))
                weights = (1.0 + magnitude * (freq_sum.float() / normalizer)).clamp_min(0.0)
                weights[0, 0] = 1.0
                component = component * weights.view(1, 1, 1, block_height, block_width)
            coeff[component_index] = component
        return coeff

    def _apply_op(self, coeff, op_name, magnitude):
        if op_name == "Identity":
            return coeff
        if op_name == "TranslateX":
            return self._translate(coeff, magnitude, direction="W")
        if op_name == "TranslateY":
            return self._translate(coeff, magnitude, direction="H")
        if op_name == "Cutout":
            return self._cutout(coeff, magnitude)
        if op_name == "Grayscale":
            if len(coeff) > 1:
                coeff[1] = coeff[1] * 0
            return coeff
        if op_name == "ChromaDrop":
            if len(coeff) > 1:
                chroma_index = 0 if torch.rand(1).item() > 0.5 else 1
                coeff[1][chroma_index] *= 0
            return coeff
        if op_name == "BlockDropout":
            return self._block_dropout(coeff, magnitude)
        if op_name == "FreqBandDropout":
            return self._freq_band_dropout(coeff, magnitude)
        if op_name == "ACQuantNoise":
            return self._ac_quant_noise(coeff, magnitude)
        if op_name == "Brightness":
            return self._brightness(coeff, magnitude)
        if op_name == "Contrast":
            coeff[0] = self._scale_dc(coeff[0], max(0.0, 1.0 + magnitude))
            return coeff
        if op_name == "Color":
            if len(coeff) > 1:
                coeff[1] = coeff[1] * max(0.0, 1.0 + magnitude)
            return coeff
        if op_name == "Sharpness":
            return self._scale_freq(coeff, magnitude, mode="sharp")
        if op_name == "MidfreqAug":
            return self._scale_freq(coeff, magnitude, mode="mid")
        if op_name == "FreqEnhance":
            return self._scale_freq(coeff, magnitude, mode="ac")
        raise ValueError(f"The provided QuantDCT operator {op_name} is not recognized.")

    def forward(self, coeff: tuple) -> Tensor:
        if len(self.ops_list) == 0 or self.num_ops <= 0:
            return coeff
        coeff = dct_prep(coeff, mode='start')
        dtypes = [component.dtype for component in coeff]
        _, height, width, _, _ = coeff[0].shape
        op_meta = self._augmentation_space(self.num_magnitude_bins, (height, width))
        ops_list = self.ops_list.copy()
        chroma_ops = {"Grayscale", "Color", "ChromaDrop"}
        for _ in range(self.num_ops):
            if len(ops_list) == 0:
                break
            op_index = int(torch.randint(len(ops_list), (1,)).item())
            op_name = ops_list[op_index]
            if op_name in chroma_ops:
                if op_name == "Grayscale":
                    ops_list = list(set(ops_list).difference(chroma_ops))
                else:
                    ops_list = list(set(ops_list).difference({"Grayscale"}))
            magnitudes, signed = op_meta[op_name]
            if magnitudes.ndim > 0:
                magnitude_index = max(0, min(int(self.magnitude), magnitudes.numel() - 1))
                magnitude = float(magnitudes[magnitude_index].item())
            else:
                magnitude = float(magnitudes.item())
            if signed and torch.randint(2, (1,)).item():
                magnitude *= -1.0
            coeff = self._apply_op(coeff, op_name, magnitude)
            coeff = self._finalize(coeff, dtypes)
        return dct_prep(coeff, mode='end')

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"num_ops={self.num_ops}"
            f", magnitude={self.magnitude}"
            f", num_magnitude_bins={self.num_magnitude_bins}"
            f", max_abs={self.max_abs}"
            f", ops_list={self.ops_list}"
            f")"
        )

def _apply_op_dct(
    coeff: Tensor, op_name: str, magnitude: float,
    pad, conv_Ls: list, conv_Ms: list,
):
    # coeff: list of coefficients. [Y, cbcr] expected
    if op_name == "ShearX":
        for i in range(len(coeff)):
            coeff[i], conv_Ls[i], conv_Ms[i] = dops.shear_block(
                coeff[i], deg_x=magnitude, conv_L=conv_Ls[i], conv_M=conv_Ms[i], pad=pad)
    elif op_name == "ShearY":
        for i in range(len(coeff)):
            coeff[i], conv_Ls[i], conv_Ms[i] = dops.shear_block(
                coeff[i], deg_y=magnitude, conv_L=conv_Ls[i], conv_M=conv_Ms[i], pad=pad)
    elif op_name == "TranslateX":
        transblocks = int(magnitude - (magnitude % 2)) # make it even
        coeff[0] = dops.translate_dct(coeff[0], transblocks, direction='W') # Y
        coeff[1] = dops.translate_dct(coeff[1], transblocks//2, direction='W') # cbcr
    elif op_name == "TranslateY":
        transblocks = int(magnitude - (magnitude % 2)) # make it even
        coeff[0] = dops.translate_dct(coeff[0], transblocks, direction='H') # Y
        coeff[1] = dops.translate_dct(coeff[1], transblocks//2, direction='H') # cbcr
    elif op_name == "Rotate":
        for i in range(len(coeff)):
            coeff[i], conv_Ls[i], conv_Ms[i] = dops.rotate_block(
                coeff[i], degrees=magnitude, conv_L=conv_Ls[i], conv_M=conv_Ms[i], pad=pad)
    elif op_name == "Brightness":
        coeff[0] = dops.brightness_dct(coeff[0], 1.0 + magnitude) # use it only on Y
    elif op_name == "Color":
        coeff[1] = dops.contrast_dct(coeff[1], 1.0 + magnitude) # use it only on cbcr
    elif op_name == "Contrast":
        coeff[0] = dops.contrast_dct(coeff[0], 1.0 + magnitude) # use it only on Y
    elif op_name == "Sharpness": # only sharpen/blur brightness
        coeff[0] = dops.sharpblur_dct(coeff[0], magnitude)
    elif op_name == "Posterize":
        for i in range(len(coeff)):
            coeff[i] = dops.posterize_dct(coeff[i], bitoffset=int(magnitude))
    elif op_name == "Solarize":
        coeff[0], dcThresholdMask = dops.solarize_dct(coeff[0], magnitude) 
        coeff[1], _ = dops.solarize_dct(coeff[1], magnitude, dcThresholdMask[:,::2,::2].repeat(2,1,1))
    elif op_name == "AutoContrast":
        coeff[0] = dops.autocontrast_dct(coeff[0]) # only on Y
    elif op_name == "Equalize":
        coeff[0] = dops.equalize_dct(coeff[0]) # only on Y
    elif op_name == "Invert":
        for i in range(len(coeff)):
            coeff[i] = dops.invert_dct(coeff[i])
    elif op_name == "Identity":
        pass 
    elif op_name == 'Cutout': # added
        cutout_size = round(magnitude)
        cutout_size = int(cutout_size - (cutout_size%2)) # make it even
        coeff[0], ccH, ccW = dops.cutout_dct(coeff[0], cutout_size, replace=0) # cutout_center_H/W
        coeff[1], _, _ = dops.cutout_dct(coeff[1], cutout_size//2, 0, ccH//2, ccW//2)
    elif op_name == "SolarizeAdd": # added
        coeff[0], _ = dops.solarize_add_dct(coeff[0], int(magnitude), threshold=0) # better to use it only on Y
    elif op_name == "Rotate90": # New
        coeff[0] = dops.rotate_dct_90deg(coeff[0], rotate=magnitude)
        coeff[1] = dops.rotate_dct_90deg(coeff[1], rotate=magnitude)
    elif op_name == "AutoSaturation": # New
        coeff[1] = dops.autocontrast_dct(coeff[1]) # only on CbCr
    elif op_name == "Grayscale": # New
        coeff[1] *= 0 # zero-out chroma channel
    elif op_name == "MidfreqAug":
        coeff[0] = dops.midfreqaug_dct(coeff[0], magnitude)
    elif op_name == "FreqEnhance": # New
        coeff[0] = dops.freq_enhance_dct(coeff[0], 1.0 + magnitude)
        coeff[1] = dops.freq_enhance_dct(coeff[1], 1.0 + magnitude)
    elif op_name == "ChromaDrop": # New 2
        if torch.rand(1).item() > 0.5:
            coeff[1][0] *= 0 # drop Cb
        else:
            coeff[1][1] *= 0 # drop Cr
    else:
        raise ValueError(f"The provided operator {op_name} is not recognized.")
    
    for i in range(len(coeff)):
        coeff[i] = coeff[i].clamp(min=-2**10, max=2**10-8).contiguous() # clamp to 8-bit max DCT coeff range after each augmentation
    return coeff


class RandAugment_dct(torch.nn.Module):
    r"""Random data augmentation method for DCT coefficients based on
    `"RandAugment: Practical automated data augmentation with a reduced search space"
    <https://arxiv.org/abs/1909.13719>`_.

    DCT coefficients are expected to have dtype torch.int16 and shape
    [..., C, H, W, KH, KW], where ... means an arbitrary number of leading dimensions.

    Args:
        num_ops (int): Number of augmentation transformations to apply sequentially.
        magnitude (int): Magnitude for all the transformations.
        num_magnitude_bins (int): The number of different magnitude values.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.NEAREST``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` are supported.
        fill (sequence or number, optional): Pixel fill value for the area outside the transformed
            image. If given a number, the value is used for all bands respectively. (Not used, defaults to 0 on all ops)
        pad (bool): If true, use padding on rotation/shearing operations
        ops_list: List of operation names in string to use during training (Default: None -- choose all augmentations except identity)
    """

    def __init__(
        self,
        num_ops: int = 2,
        magnitude: int = 10,
        num_magnitude_bins: int = 11,
        pad = 2**0.5,
        ops_list = None,
    ) -> None:
        super().__init__()
        self.num_ops = num_ops
        self.magnitude = magnitude
        self.num_magnitude_bins = num_magnitude_bins
        self.pad = pad
        self.conv_Ls = [None, None] # conv_L for Y and cbcr (conversion matrix)
        self.conv_Ms = [None, None] # conv_M for Y and cbcr (conversion matrix)
        if ops_list == None:
            self.ops_list = ["AutoContrast", "Equalize", "Invert", "Rotate", "Posterize", "Solarize", "SolarizeAdd", "Color", "Contrast", "Brightness",
                            "Sharpness", "ShearX", "ShearY", "Cutout", "TranslateX", "TranslateY"] # this is the default ops list for RandAugment (original)
        else:
            self.ops_list = ops_list

    def _augmentation_space(self, num_bins: int, image_size: Tuple[int, int]) -> Dict[str, Tuple[Tensor, bool]]:
        return {
            # op_name: (magnitudes, signed)
            "Identity": (torch.tensor(0.0), False),
            "AutoContrast": (torch.tensor(0.0), False),
            "Equalize": (torch.tensor(0.0), False),
            "Invert": (torch.tensor(0.0), False), # added
            "Rotate": (torch.linspace(0.0, 30.0, num_bins), True),
            "Posterize": (torch.linspace(0.0, 5.0, num_bins).round().int(), False),
            "Solarize": (torch.linspace(818, -818, num_bins), False),
            "SolarizeAdd": (torch.linspace(0, 883, num_bins), False), # added (110/255 = 883/2048)
            "Color": (torch.linspace(0.0, 0.9, num_bins), True),
            "Contrast": (torch.linspace(0.0, 0.9, num_bins), True),
            "Brightness": (torch.linspace(0.0, 0.9, num_bins), True),
            "Sharpness": (torch.linspace(0.0, 0.9, num_bins), True), # DCT sharpen/blur using linear kernel
            "ShearX": (torch.linspace(0.0, 17., num_bins), True), # degrees(arctan(0.3)) = 16.7 ~= 17
            "ShearY": (torch.linspace(0.0, 17., num_bins), True),
            "Cutout": (torch.linspace(0, 6, num_bins), False), #added # 40/224 = 5/28 --- assuming 8x8 dct block (increased to 6 for even cutout btw Y/cbcr)
            "TranslateX": (torch.linspace(0.0, 150.0 / 336.0 * image_size[1], num_bins), True),
            "TranslateY": (torch.linspace(0.0, 150.0 / 336.0 * image_size[0], num_bins), True),
            "Rotate90": (torch.tensor(1), True), # New -- rotates 90 degrees (clockwise or counter-clockwise)
            "AutoSaturation": (torch.tensor(0.0), False), # New -- applies AutoContrast on Chroma channel
            "Grayscale": (torch.tensor(0.0), False), # New -- applies grayscale (chroma *= 0)
            "MidfreqAug": (torch.linspace(0.0, 0.9, num_bins), True), # New -- sharpen/blurs the mid frequency using gaussian kernel
            "FreqEnhance": (torch.linspace(0.0, 0.9, num_bins), True), # New -- multiplies all frequency component by this value
            "ChromaDrop": (torch.tensor(0.0), False), # New 2 -- drop Cb or Cr channel only
        }


    def forward(self, coeff: tuple) -> Tensor:
        """
            coeff (Tuple of tensor, (Y, cbcr) or (Y,) or (cbcr,)): DCT Coefficient to be transformed.

        Returns:
            Tensor: Transformed coefficient.
        """
        if len(self.ops_list)==0: # if empty ops_list
            return coeff

        _, height, width, _, _ = coeff[0].shape
        coeff = dops.dct_prep(coeff, mode='start')
        for i in range(len(coeff)):
            coeff[i] = coeff[i].clamp(min=-2**10, max=2**10-8) # clamp to 8-bit max DCT coeff range (they can possibly go over those after resize/flip/rotate/etc augmentation)
        ops_list = self.ops_list.copy() # local ops_list
        op_meta = self._augmentation_space(self.num_magnitude_bins, (height, width))
        chromas = {"Grayscale", "Color", "AutoSaturation", "ChromaDrop"}
        for _ in range(self.num_ops):
            op_index = int(torch.randint(len(ops_list), (1,)).item()) # choose from augmentations in ops_list (so you can choose to not use certain augmentations during training)
            op_name = ops_list[op_index] # run that op
            if op_name in chromas:
                if op_name=="Grayscale":
                    ops_list = list(set(ops_list).difference(chromas)) # don't use chroma augmentation if grayscale is applied
                else:
                    ops_list = list(set(ops_list).difference({"Grayscale"})) # don't use grayscale if chroma augmentation is applied
            magnitudes, signed = op_meta[op_name]
            magnitude = float(magnitudes[self.magnitude].item()) if magnitudes.ndim > 0 else magnitudes.item()
            if signed and torch.randint(2, (1,)):
                magnitude *= -1.0
            coeff = _apply_op_dct(coeff, op_name, magnitude, pad=self.pad, conv_Ls=self.conv_Ls, conv_Ms=self.conv_Ms)
            assert coeff[0].dtype != torch.float32, f"Coeff dtype should not be float32. Current augmentation: {op_name}, dtype: {coeff[0].dtype}" # assuming uint8 dtype

        return dops.dct_prep(coeff, mode='end')


    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"num_ops={self.num_ops}"
            f", magnitude={self.magnitude}"
            f", num_magnitude_bins={self.num_magnitude_bins}"
            f")"
        )
        return s

class ycbcr_to_rgb(torch.nn.Module):
    """
    Converts YCbCr color scheme to RGB
    """
    def __init__(
        self, 
    ):
        super().__init__()
    
    def forward(self, coeff):
        """
        Args:
            coeff (Tensor or tuple): dct coefficient with five channels in (c, h, w, kh, kw)
                                    if tuple, (Y, cbcr) is expected.

        Returns:
            Tensor: Converted YCbCr to RGB data
        """
        Y, cbcr = coeff
        assert Y.dtype == torch.float32 and cbcr.dtype == torch.float32, f"Y and CbCr dtype should be torch.float32. Current:{Y.dtype}, {cbcr.dtype}"

        ### Convert using JPEG encoder ###
        _, H, W, KH, KW = Y.shape
        _, CH, CW, _, _ = cbcr.shape
        dim_inferred = torch.tensor([[H*KH, W*KW], [CH*KH, CW*KW], [CH*KH, CW*KW]], dtype=torch.int32)
        quant_100 = 2*torch.ones((3,8,8), dtype=torch.int16)
        RGBimg = dm.decode_coeff(
            dim_inferred, quant_100, 
            (Y/2).round().to(torch.int16).clamp(min=-1024, max=1016).contiguous(), 
            (cbcr/2).round().to(torch.int16).clamp(min=-1024, max=1016).contiguous()
            )
        return RGBimg
        

class rgb_to_dct(torch.nn.Module):
    """
    Converts RGB to YCbCr DCT
    """
    def __init__(
        self, 
    ):
        super().__init__()
    
    def forward(self, img):
        """
        Args:
            img (Tensor or PIL input image)

        Returns:
            coeff (Tensor): Converted RGB to YCbCr DCT data
        """
        ### Convert using JPEG encoder ###
        if F._is_pil_image(img):
            img = F.pil_to_tensor(img)
        _, _, Y, cbcr = dm.quantize_at_quality(img, 100)

        return (Y, cbcr)
