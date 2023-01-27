import numpy as np
import galsim
import ngmix


def sim_func(*, gal, g1, g2, seed, noise):
    # this is an RNG you can use to draw random numbers as needed
    # always use this RNG and not np.random directly
    # doing this makes sure the code is reproducible

    rng = np.random.RandomState(seed=seed)

    # make a Gaussian object in galsim with a fwhm of `psf_fwhm`
    psf = galsim.Moffat(fwhm=0.7, beta=2.5)
    # psf = galsim.Gaussian(fwhm=0.7)

    # apply the input shear `g1`, `g2` to the galaxy `gal`

    ### draw shear randomly scale=0.3 for shear
    sheared_gal = gal.shear(g1=g1, g2=g2)

    # here we are going to apply a random shift to the object's center
    # dx, dy = 2.0 * (rng.uniform(size=2) - 0.5) * 0.2
    # sheared_gal = sheared_gal.shift(dx, dy)

    # convolve the sheared galaxy with the psf
    obj = galsim.Convolve(sheared_gal, psf)

    # render the object and the PSF on odd sized images of 53 pixels on a side with
    # a pixel scale of 0.2
    obj_im = obj.drawImage(scale=0.2, nx=53, ny=53)
    psf_im = psf.drawImage(scale=0.2, nx=53, ny=53)

    cen = (53 - 1) / 2

    # nse = (
    #     np.sqrt(np.sum(
    #         galsim.Convolve([
    #             psf,
    #             galsim.Exponential(half_light_radius=0.5),
    #         ]).drawImage(scale=0.25).array**2)
    #     )
    #     / snr
    # )

    #TODO: noise from WL deblending

    nse_sd = noise

    nse = rng.normal(size=obj_im.array.shape, scale=nse_sd)
    nse_im = rng.normal(size=obj_im.array.shape, scale=nse_sd)

    jac = ngmix.jacobian.DiagonalJacobian(scale=0.2,
                                          row=cen / 0.2,
                                          col=cen / 0.2)
    psf_jac = ngmix.jacobian.DiagonalJacobian(scale=0.2, row=cen, col=cen)

    # Transformation between pixel and tangent uv coordinate. It has off diag terms if CCD is rotated.
    # in real data we care about off diag terms

    # we have to add a little noise to the PSf to make things stable
    target_psf_s2n = 500.0
    target_psf_noise = np.sqrt(np.sum(psf_im.array**2)) / target_psf_s2n

    psf_obs = ngmix.Observation(
        image=psf_im.array,
        weight=np.ones_like(psf_im.array) / target_psf_noise**2,
        jacobian=psf_jac,
    )

    # here we build the final observation
    obj_obs = ngmix.Observation(
        image=obj_im.array + nse,
        noise=nse_im,
        weight=np.ones_like(nse_im) / nse_sd**2,
        jacobian=psf_jac,
        bmask=np.zeros_like(nse_im, dtype=np.int32),
        ormask=np.zeros_like(nse_im, dtype=np.int32),
        psf=psf_obs,
    )

    return obj_obs