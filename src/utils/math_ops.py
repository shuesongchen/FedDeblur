import torch
import numpy as np
import random
import torch.backends.cudnn as cudnn


def setup_all(seed=0, gpu_id=0):
    torch.cuda.empty_cache()
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

    torch.set_default_device(f"cuda:{gpu_id}")
    device = torch.device(f"cuda:{gpu_id}")

    return device


def Dx(u):
    """Computes the backward finite difference in the x-direction (axis=1) with circular boundary conditions."""
    return u - torch.roll(u, 1, dims=1)


def Dxt(u):
    """Computes the transpose of the Dx operator, which is the negative forward finite difference in the x-direction (axis=1) with circular boundary conditions."""
    return u - torch.roll(u, -1, dims=1)


def Dy(u):
    """Computes the backward finite difference in the y-direction (axis=0) with circular boundary conditions."""
    return u - torch.roll(u, 1, dims=0)


def Dyt(u):
    """Computes the transpose of the Dy operator, which is the negative forward finite difference in the y-direction (axis=0) with circular boundary conditions."""
    return u - torch.roll(u, -1, dims=0)


def shrink_soft_2(x, y, lambda_val):
    """
    Performs soft thresholding on a vector field (x, y), equivalent to the MATLAB shrink2 function.
    Handles both 2D (grayscale) and 3D (color/hyperspectral/video) data.
    """
    if x.ndim == 2:
        # 2D case: grayscale images
        s = torch.maximum(torch.sqrt(x**2 + y**2), torch.tensor(1e-6, device=x.device))
    else:
        # 3D case: color images
        s = torch.maximum(
            torch.sqrt(torch.sum(x**2 + y**2, dim=2)),
            torch.tensor(1e-6, device=x.device),
        )
        s = s.unsqueeze(2).repeat(1, 1, x.shape[2])

    xs = torch.maximum(torch.tensor(0.0, device=x.device), s - lambda_val) * x / s
    ys = torch.maximum(torch.tensor(0.0, device=x.device), s - lambda_val) * y / s
    return xs, ys


def generate_param_list(p_star, r=1.1, m=5):
    params = [p_star / (r**i) for i in range(m, 0, -1)]
    params += [p_star]
    params += [p_star * (r**i) for i in range(1, m + 1)]
    return params


# https://github.com/aboucaud/pypher/blob/master/pypher/pypher.py


def psf2otf(psf, shape):
    """
    Convert point-spread function to optical transfer function.

    Compute the Fast Fourier Transform (FFT) of the point-spread
    function (PSF) array and creates the optical transfer function (OTF)
    array that is not influenced by the PSF off-centering.
    By default, the OTF array is the same size as the PSF array.

    To ensure that the OTF is not altered due to PSF off-centering, PSF2OTF
    post-pads the PSF array (down or to the right) with zeros to match
    dimensions specified in OUTSIZE, then circularly shifts the values of
    the PSF array up (or to the left) until the central pixel reaches (1,1)
    position.

    Parameters
    ----------
    psf : `torch.Tensor`
        PSF array
    shape : int
        Output shape of the OTF array

    Returns
    -------
    otf : `torch.Tensor`
        OTF array

    Notes
    -----
    Adapted from MATLAB psf2otf function
    Arrays of higher dimension that 2D are also supported

    """
    if torch.all(psf == 0):
        return torch.zeros_like(psf)

    inshape = psf.shape
    # Pad the PSF to outsize
    psf = zero_pad(psf, shape, position="corner")

    # Circularly shift OTF so that the 'center' of the PSF is
    # [0,0] element of the array
    for axis, axis_size in enumerate(inshape):
        psf = torch.roll(psf, -int(axis_size / 2), dims=axis)

    # Compute the OTF
    otf = torch.fft.fftn(psf)

    # Estimate the rough number of operations involved in the FFT
    # and discard the PSF imaginary part if within roundoff error
    n_ops = torch.sum(
        # torch.tensor(
        psf.numel()
        * torch.log2(torch.tensor(psf.shape, dtype=torch.float)).sum()
        # )
    )

    # Check if imaginary part is negligible
    if torch.max(torch.abs(torch.imag(otf))) <= n_ops * torch.finfo(torch.float).eps:
        otf = torch.real(otf)

    return otf


def zero_pad(image, shape, position="corner"):
    """
    Extends image to a certain size with zeros

    Parameters
    ----------
    image: real 2d `torch.Tensor`
        Input image
    shape: tuple of int
        Desired output shape of the image
    position : str, optional
        The position of the input image in the output one:
            * 'corner'
                top-left corner (default)
            * 'center'
                centered

    Returns
    -------
    padded_img: real `torch.Tensor`
        The zero-padded image

    """
    shape = torch.tensor(shape, dtype=torch.int32)
    imshape = torch.tensor(image.shape, dtype=torch.int32)

    if torch.all(imshape == shape):
        return image

    if torch.any(shape <= 0):
        raise ValueError("ZERO_PAD: null or negative shape given")

    dshape = shape - imshape
    if torch.any(dshape < 0):
        raise ValueError("ZERO_PAD: target size smaller than source one")

    pad_img = torch.zeros(shape.tolist(), dtype=image.dtype, device=image.device)

    if position == "center":
        if torch.any(dshape % 2 != 0):
            raise ValueError(
                "ZERO_PAD: source and target shapes " "have different parity."
            )
        offx, offy = dshape // 2
    else:
        offx, offy = (0, 0)

    pad_img[offx : offx + imshape[0], offy : offy + imshape[1]] = image

    return pad_img


def imfilter(image, kernel, mode="circular", conv=True):
    """
    Image filtering function with circular boundary conditions

    Implements image convolution using FFT, automatically supports circular boundary conditions,
    input and output images have the same size.
    Specifically designed for torchvision.io.read_image().float() output format.

    Parameters
    ----------
    image : torch.Tensor
        Input image, format (C, H, W):
        - Color image: (3, H, W)
        - Grayscale image: (1, H, W)
        From torchvision.io.read_image(path).float()
    kernel : torch.Tensor
        Convolution kernel, shape (Kh, Kw)
    mode : str, optional
        Boundary condition, currently only supports "circular" (default)
    conv : bool, optional
        Whether to perform convolution (True, default) or correlation (False)
        Convolution flips the kernel, correlation does not

    Returns
    -------
    output : torch.Tensor
        Filtered image, same shape as input image (C, H, W)
    """

    if mode != "circular":
        raise NotImplementedError(
            "Currently only supports circular boundary conditions"
        )

    # Check input format
    if image.ndim != 3:
        raise ValueError(f"Input image must be 3D tensor (C, H, W), got {image.ndim}D")

    channels, height, width = image.shape
    kernel_h, kernel_w = kernel.shape

    # conv vs correlation difference
    if not conv:
        kernel = torch.flip(kernel, dims=[0, 1])

    # Ensure kernel and image are on the same device and have matching data types
    kernel = kernel.to(device=image.device, dtype=image.dtype)

    # Process each channel
    output = torch.zeros_like(image)

    for c in range(channels):
        # Get single channel image
        img_2d = image[c]  # (H, W)

        # Convert image to frequency domain
        img_fft = torch.fft.fft2(img_2d)

        # Use psf2otf to convert kernel to frequency domain representation of same size
        kernel_otf = psf2otf(kernel, (height, width))

        # Element-wise multiplication in frequency domain (convolution theorem)
        result_fft = img_fft * kernel_otf

        # Convert back to spatial domain
        result = torch.fft.ifft2(result_fft)

        # Take real part (ignore tiny imaginary part due to numerical errors)
        output[c] = torch.real(result)

    return output


# https://github.com/italogsfernandes/imagens-medicas-2/blob/master/Aulas/aula_2_-_uniform_filter/matlab_fspecial.py
# https://github.com/tkkcc/prior/blob/879a0b6c117c810776d8cc6b63720bf29f7d0cc4/util/gen_kernel.py

"""
Syntax
kernel = fspecial(type)
kernel = fspecial('average',hsize)
kernel = fspecial('disk',radius)
kernel = fspecial('gaussian',hsize,sigma)
kernel = fspecial('laplacian',alpha)
kernel = fspecial('log',hsize,sigma)
kernel = fspecial('motion',len,theta)
kernel = fspecial('prewitt')
kernel = fspecial('sobel')
kernel = fspecial('synthesis',h,w)
scipy.ndimage.convolve(image, kernel, mode="wrap")
"""


def fspecial_average(hsize=3):
    """Smoothing filter"""
    return torch.ones((hsize, hsize)) / hsize**2


def fspecial_disk(radius):
    """
    Disk filter - Circular averaging filter
    Based on MATLAB implementation from Peter Kovesi
    """
    # Calculate filter size
    size = 2 * radius + 1
    center = radius

    # Create coordinate matrices
    x, y = torch.meshgrid(
        torch.arange(-center, center + 1, dtype=torch.float),
        torch.arange(-center, center + 1, dtype=torch.float),
        indexing="ij",
    )

    # Calculate distance from center
    rad = torch.sqrt(x**2 + y**2)

    # Create circular mask
    f = (rad <= radius).float()

    # Normalize
    f = f / torch.sum(f)

    return f


def fspecial_gaussian(hsize, sigma):
    """Gaussian filter"""
    hsize = [hsize, hsize]
    siz = [(hsize[0] - 1.0) / 2.0, (hsize[1] - 1.0) / 2.0]
    std = sigma
    [x, y] = torch.meshgrid(
        torch.arange(-siz[1], siz[1] + 1, dtype=torch.float),
        torch.arange(-siz[0], siz[0] + 1, dtype=torch.float),
        indexing="ij",
    )
    arg = -(x * x + y * y) / (2 * std * std)
    h = torch.exp(arg)
    h[h < torch.finfo(torch.float32).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h = h / sumh
    return h


def fspecial_laplacian(alpha):
    """Laplacian filter"""
    alpha = max([0, min([alpha, 1])])
    h1 = alpha / (alpha + 1)
    h2 = (1 - alpha) / (alpha + 1)
    h = [[h1, h2, h1], [h2, -4 / (alpha + 1), h2], [h1, h2, h1]]
    h = torch.tensor(h, dtype=torch.float)
    return h


def fspecial_log(hsize, sigma):
    """
    Laplacian of Gaussian filter
    """
    if isinstance(hsize, (int, float)):
        hsize = [hsize, hsize]

    rows, cols = hsize[0], hsize[1]
    r2 = (rows - 1) / 2
    c2 = (cols - 1) / 2

    # Create coordinate matrices
    x, y = torch.meshgrid(
        torch.arange(-c2, c2 + 1, dtype=torch.float),
        torch.arange(-r2, r2 + 1, dtype=torch.float),
        indexing="ij",
    )
    radsqrd = x**2 + y**2

    # Laplacian of Gaussian formula
    f = (
        -1
        / (torch.pi * sigma**4)
        * (1 - radsqrd / (2 * sigma**2))
        * torch.exp(-radsqrd / (2 * sigma**2))
    )

    # Ensure 0 DC (zero mean)
    f = f - torch.mean(f)

    return f


def fspecial_motion(length, angle):
    """
    Motion blur filter - Pure PyTorch implementation

    Parameters:
    -----------
    length : float
        Length of the motion blur
    angle : float
        Angle of motion in degrees
    """
    EPS = torch.finfo(torch.float32).eps
    alpha = (angle - torch.floor(torch.tensor(angle / 180)) * 180) / 180 * torch.pi
    cosalpha = torch.cos(alpha)
    sinalpha = torch.sin(alpha)

    if cosalpha < 0:
        xsign = -1
    elif angle == 90:
        xsign = 0
    else:
        xsign = 1

    psfwdt = 1

    sx = int(torch.abs(length * cosalpha + psfwdt * xsign - length * EPS).item())
    sy = int(torch.abs(length * sinalpha + psfwdt - length * EPS).item())

    sx = max(1, sx)
    sy = max(1, sy)

    psf1 = torch.zeros((sy, sx), dtype=torch.float32)
    half = length / 2

    # Create coordinate grids for vectorized computation
    i_coords, j_coords = torch.meshgrid(
        torch.arange(sy, dtype=torch.float32),
        torch.arange(sx, dtype=torch.float32),
        indexing="ij",
    )

    # Vectorized computation
    psf_vals = i_coords * torch.abs(cosalpha) - j_coords * sinalpha
    rad = torch.sqrt(i_coords * i_coords + j_coords * j_coords)

    # Apply conditions vectorized
    mask = (rad >= half) & (torch.abs(psf_vals) <= psfwdt)

    if torch.abs(cosalpha) > EPS:  # 避免除零
        temp = half - torch.abs((j_coords + psf_vals * sinalpha) / cosalpha)
        psf_vals = torch.where(
            mask, torch.sqrt(psf_vals * psf_vals + temp * temp), psf_vals
        )

    psf1 = psfwdt + EPS - torch.abs(psf_vals)
    psf1 = torch.where(psf1 < 0, torch.zeros_like(psf1), psf1)

    # Handle different motion directions
    if 0 < angle < 90:
        psf1 = torch.fliplr(psf1)
    elif -90 < angle < 0:
        psf1 = torch.flipud(psf1)
        psf1 = torch.fliplr(psf1)
    elif angle < -90:
        psf1 = torch.flipud(psf1)

    # Normalize
    if psf1.sum() > 0:
        psf1 = psf1 / psf1.sum()

    return psf1


def fspecial_prewitt():
    """Prewitt edge filter"""
    return torch.tensor([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=torch.float)


def fspecial_sobel():
    """Sobel edge filter"""
    return torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float)


def blurkernel_synthesis(h=37, w=None):
    """
    Generate random blur kernel - Pure PyTorch implementation
    """
    w = h if w is None else w
    kdims = [h, w]
    x = randomTrajectory(200)
    k = None
    while k is None:
        k = kernelFromTrajectory(x)

    # center pad to kdims using torch.nn.functional.pad
    if k is not None:
        pad_h = (kdims[0] - k.shape[0]) // 2
        pad_w = (kdims[1] - k.shape[1]) // 2
        # torch.nn.functional.pad expects (pad_left, pad_right, pad_top, pad_bottom)
        k = torch.nn.functional.pad(
            k, (pad_w, pad_w, pad_h, pad_h), mode="constant", value=0
        )

    return k


def kernelFromTrajectory(x):
    """Generate kernel from trajectory - Pure PyTorch implementation"""
    # Convert numpy input to torch if needed
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).float()

    h = 5 - torch.log(torch.rand(1)).item() / 0.15
    h = torch.round(torch.min(torch.tensor([h, 27.0]))).int().item()
    h = h + 1 - h % 2
    w = h
    k = torch.zeros((h, w), dtype=torch.float32)

    xmin = torch.min(x[0])
    xmax = torch.max(x[0])
    ymin = torch.min(x[1])
    ymax = torch.max(x[1])

    xthr = torch.linspace(xmin, xmax, w + 1)[
        :-1
    ]  # equivalent to np.arange but more stable
    ythr = torch.linspace(ymin, ymax, h + 1)[:-1]

    # Vectorized computation for better performance
    for i in range(1, len(xthr)):
        for j in range(1, len(ythr)):
            if i < len(xthr) and j < len(ythr):
                idx = (
                    (x[0, :] >= xthr[i - 1])
                    & (x[0, :] < xthr[i])
                    & (x[1, :] >= ythr[j - 1])
                    & (x[1, :] < ythr[j])
                )
                k[i - 1, j - 1] = torch.sum(idx.float())

    if torch.sum(k) == 0:
        return None

    k = k / torch.sum(k)

    # Replace scipy.signal.convolve2d with torch convolution
    gaussian_kernel = fspecial_gaussian(3, 1)
    # Reshape for conv2d: (batch_size, in_channels, height, width)
    k_4d = k.unsqueeze(0).unsqueeze(0)
    gaussian_4d = gaussian_kernel.unsqueeze(0).unsqueeze(0)

    # Apply convolution with 'same' padding
    pad_size = gaussian_kernel.shape[0] // 2
    k_conv = torch.nn.functional.conv2d(k_4d, gaussian_4d, padding=pad_size)
    k = k_conv.squeeze(0).squeeze(0)

    k = k / torch.sum(k)
    return k


def randomTrajectory(T):
    """Generate random trajectory for blur kernel - Pure PyTorch implementation"""
    x = torch.zeros((3, T), dtype=torch.float32)
    v = torch.randn(3, T, dtype=torch.float32)
    r = torch.zeros((3, T), dtype=torch.float32)
    trv = 1.0
    trr = 2 * torch.pi / T

    for t in range(1, T):
        F_rot = torch.randn(3, dtype=torch.float32) / (t + 1) + r[:, t - 1]
        F_trans = torch.randn(3, dtype=torch.float32) / (t + 1)
        r[:, t] = r[:, t - 1] + trr * F_rot
        v[:, t] = v[:, t - 1] + trv * F_trans
        st = v[:, t]
        st = rot3D(st, r[:, t])
        x[:, t] = x[:, t - 1] + st

    return x


def rot3D(x, r):
    """3D rotation - Pure PyTorch implementation"""
    # Rotation matrices using PyTorch
    cos_r0, sin_r0 = torch.cos(r[0]), torch.sin(r[0])
    cos_r1, sin_r1 = torch.cos(r[1]), torch.sin(r[1])
    cos_r2, sin_r2 = torch.cos(r[2]), torch.sin(r[2])

    Rx = torch.tensor(
        [[1, 0, 0], [0, cos_r0, -sin_r0], [0, sin_r0, cos_r0]], dtype=torch.float32
    )

    Ry = torch.tensor(
        [[cos_r1, 0, sin_r1], [0, 1, 0], [-sin_r1, 0, cos_r1]], dtype=torch.float32
    )

    Rz = torch.tensor(
        [[cos_r2, -sin_r2, 0], [sin_r2, cos_r2, 0], [0, 0, 1]], dtype=torch.float32
    )

    R = torch.mm(torch.mm(Rz, Ry), Rx)
    x = torch.mv(R, x)
    return x


def fspecial(filter_type, *args, **kwargs):
    """
    Main fspecial function - mimics MATLAB's fspecial
    """

    if filter_type == "average":
        return fspecial_average(*args, **kwargs)
    elif filter_type == "disk":
        return fspecial_disk(*args, **kwargs)
    elif filter_type == "gaussian":
        return fspecial_gaussian(*args, **kwargs)
    elif filter_type == "laplacian":
        return fspecial_laplacian(*args, **kwargs)
    elif filter_type == "log":
        return fspecial_log(*args, **kwargs)
    elif filter_type == "motion":
        return fspecial_motion(*args, **kwargs)
    elif filter_type == "prewitt":
        return fspecial_prewitt(*args, **kwargs)
    elif filter_type == "sobel":
        return fspecial_sobel(*args, **kwargs)
    elif filter_type == "synthesis":
        return blurkernel_synthesis(*args, **kwargs)
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")
