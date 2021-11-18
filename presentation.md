---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.1
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

```{code-cell} ipython3
---
slideshow:
  slide_type: '-'
---
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from IPython.display import YouTubeVideo, Markdown, Code
%matplotlib notebook
```

+++ {"slideshow": {"slide_type": "slide"}}

# NumPy: A look at the past, present, and future of array computation

[Ross Barnowski](https://bids.berkeley.edu/people/ross-barnowski) `rossbar@berkeley.edu` | [@rossbar](https://github.com/rossbar) on GitHub

AMMI NumPy "Vitual Workshop" | 5/2/2020

+++ {"slideshow": {"slide_type": "slide"}}

# Part I - Overview

+++ {"slideshow": {"slide_type": "slide"}}

# What is NumPy?

> *NumPy is the fundamental package for scientific computing with Python*
> 
>  [numpy.org](https://numpy.org/)

+++ {"slideshow": {"slide_type": "fragment"}}

Strong stuff.

+++ {"slideshow": {"slide_type": "subslide"}}

<center>

## The scientific Python ecosystem

<img src="images/scipy_ecosystem.png" alt="scientific_python_ecosystem" width=40%/>

</center>

Image credit: Jarrod Millman et. al. - Upcoming NumPy Paper ([preprint here](https://github.com/bids-numpy/numpy-paper))

+++ {"slideshow": {"slide_type": "slide"}}

# A bit of history

 - **Mid 90's/Early 00's**: desire for high-performance numerical computation in Python eventually culminates in the [`Numeric`](https://numpy.org/_downloads/768fa66c250a0335ad3a6a30fae48e34/numeric-manual.pdf) library
 - Early adopters included the [Space Telescope Science Institute (STScI)](http://www.stsci.edu/) who developed another array computation package to better suit their needs: `NumArray`.
 - **2005** The best ideas from `Numeric` and `NumArray` were combined in the development of a new library, `NumPy`
   * This work was largely done by [Travis Oliphant](https://github.com/teoliphant), then an assistant professor at BYU
 - **2006** NumPy v1.0 released in October
 
[NumPy Development History](https://github.com/numpy/numpy/graphs/contributors)

+++ {"slideshow": {"slide_type": "slide"}}

# What does NumPy provide?

 - `ndarray`: A generic, n-dimensional array data structure
 - Sophisticated machinery for operating on array data
   * Powerful indexing
   * Built-in, array-aware operations
   * Vectorization and broadcasting
 - All features exposed by a concise, expressive syntax
 - Language extension/integration (C-API, `f2py`) and interoperability
   * [Array API](https://numpy.org/doc/1.17/reference/c-api.array.html) for accessing/extending array functionality
   * Protocols for replicating the NumPy interface (stay tuned...)

+++ {"slideshow": {"slide_type": "subslide"}}

## What else?

`numpy` also includes tools for common scientific/numerical tasks:
   * Random number generation (`np.random`)
   * Fourier analysis (`np.fft`)
   * Linear algebra (`np.linalg`)
   * Polynomial expressions (`np.polynomial`)

+++ {"slideshow": {"slide_type": "subslide"}}

### The `scipy` package includes modules with the same name? What's the deal?

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
import scipy, scipy.linalg
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
print(scipy.random) # scipy.stats
scipy.random is np.random 
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
print(scipy.fft)
scipy.fft is np.fft
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
print(scipy.linalg)
scipy.linalg is np.linalg
```

+++ {"slideshow": {"slide_type": "fragment"}}

## A useful analogy...

<center><img src="images/tool_analogy.png" alt="socket set analogy"/></center>

 - E.g. see [this quick comparison](https://numpy.org/devdocs/reference/routines.linalg.html) of the `numpy` and `scipy` `linalg` modules.

+++ {"slideshow": {"slide_type": "slide"}}

# Where is NumPy used?

+++ {"slideshow": {"slide_type": "slide"}}

# At a glance

<center><img src="images/NumPy_info3.jpg" alt="NumPy Overview Infographic" width=1152 height=658/></center>

**See also**: [Anaconda 2019 Year in Review](https://www.anaconda.com/2019-year-in-review/),  [NumPy PyPI Stats page](https://pypistats.org/packages/numpy)

+++ {"slideshow": {"slide_type": "slide"}}

# Part II - NumPy in the wild

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
# Code example: github graphql query for top starred projects with numpy as a dependency
```

+++ {"slideshow": {"slide_type": "skip"}}

## Neuroimaging Analysis

Like much of the scientific python ecosystem, [nipy](https://nipy.org/) relies on `np.ndarray` as the fundamental structure for neuroimaging data.

The following example is adapted from [Machine learning for neuroimaging with scikit learn](https://www.frontiersin.org/articles/10.3389/fninf.2014.00014/full). The dataset used comes from the [nilearn data](https://www.nitrc.org/frs/?group_id=728).

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
import nibabel   # package for loading/saving neuroimaging data
bg_img = nibabel.load('data/bg.nii.gz')
bg = bg_img.get_fdata()
type(bg)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
# Create activation map by thresholding the data
act_thresh = 6000
act = bg.copy()
# Set "unactivated" voxels to NaN for visualization
act[act <= act_thresh] = np.nan
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
# imshow kwargs
imshow_opts = {
    "origin" : "lower",
    "interpolation" : "nearest"
}

# Axial slice of activation map overlay
plt.imshow(bg[...,10].T, cmap="gray");             # Background
plt.imshow(act[...,10].T, cmap="plasma");          # Activation map
plt.axis('off');
```

+++ {"slideshow": {"slide_type": "skip"}}

Interested in neuroimaging? Check out [openneuro.org](https://openneuro.org/) for curated data sets from published neuroimaging studies.

+++ {"slideshow": {"slide_type": "skip"}}

## Detecting gravitational wave signature of black hole and neutron star mergers

<center><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/d/db/LIGO_measurement_of_gravitational_waves.svg/710px-LIGO_measurement_of_gravitational_waves.svg.png" alt="CBC Chirp"></center>

From [Wikipedia](https://en.wikipedia.org/wiki/First_observation_of_gravitational_waves)


```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
YouTubeVideo('I_88S8DWbcU', autoplay=1, loop=1, playlist='I_88S8DWbcU')
```

+++ {"slideshow": {"slide_type": "skip"}}

[PyCBC](https://pycbc.org/) is the toolkit used to analyze data from gravitational wave observatories like [LIGO](https://www.ligo.caltech.edu/) and [Virgo](http://www.virgo-gw.eu/).

+++ {"slideshow": {"slide_type": "skip"}}

The [PyCBC tutorials](https://github.com/gwastro/PyCBC-Tutorials) have some really cool examples - let's recreate the "chirp" from [first ever direct detection of gravitational waves](https://en.wikipedia.org/wiki/First_observation_of_gravitational_waves) that resulted from two black holes merging. For more info, see [the second PyCBC tutorial](https://colab.research.google.com/github/gwastro/pycbc-tutorials/blob/master/tutorial/2_VisualizationSignalProcessing.ipynb).

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
import pycbc
from pycbc import catalog

merger_data = catalog.Merger('GW150914')
# Though the catalog includes data from multiple observatories,
# let's focus on just one
ligo_data = merger_data.strain('L1')
type(ligo_data)
```

+++ {"slideshow": {"slide_type": "skip"}}

`pycbc` has its own (quite extensive) API that uses `numpy` and `scipy` under the hood

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
print(type(ligo_data._data))
```

+++ {"slideshow": {"slide_type": "skip"}}

To re-create the "chirp" we have to do some analysis on the raw data. 

Let's start by applying a simple band-pass filter. This is simpler than the analysis method [used in the official pycbc tutorial](https://colab.research.google.com/github/gwastro/pycbc-tutorials/blob/master/tutorial/2_VisualizationSignalProcessing.ipynb), but works suprisingly well!

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
# Apply a bandpass filter to the data
res = ligo_data.highpass_fir(20, 512).lowpass_fir(350, 512)
```

+++ {"slideshow": {"slide_type": "skip"}}

`pycbc` relies on tools in `scipy.signal` to conduct the frequency analysis.

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
pycbc.filter.lowpass_fir??
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
pycbc.filter.fir_zero_filter??
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
from scipy.signal import lfilter
lfilter?
```

+++ {"slideshow": {"slide_type": "skip"}}

Let's take a look at the results of our filter analysis...

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
time_of_merger = merger_data.time

# Look 500 msec-worth of data around the merger time
roi = res.time_slice(time_of_merger - 0.25, time_of_merger + 0.25)

# Similar to a spectrogram with more sophisticated, irregular sampling
times, freqs, power = roi.qtransform(
    delta_t=0.001,
    logfsteps=100,
    qrange=(8, 8),
    frange=(30, 512),
)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
fig, ax = plt.subplots(figsize=(6,3))
ax.pcolormesh(times, freqs, power**0.5)
ax.set_yscale('log')
```

+++ {"slideshow": {"slide_type": "slide"}}

# Generating the first ever direct image of a black hole

On April 10th 2019, the [Event Horizon Telescope](https://eventhorizontelescope.org/) collaboration released the [first ever image of a black hole](https://eventhorizontelescope.org/press-release-april-10-2019-astronomers-capture-first-image-black-hole):

<center><img src="https://static.projects.iq.harvard.edu/files/styles/os_files_xlarge/public/eht/files/20190410-78m-800x466.png?m=1554877319&itok=ryK319ed" alt="EHT_M87_04-10-19"/></center>

Image source: The [official blog post](https://eventhorizontelescope.org/press-release-april-10-2019-astronomers-capture-first-image-black-hole) from the EHT collaboration announcing the result

+++ {"slideshow": {"slide_type": "subslide"}}

The data and analysis pipeline are *way* too complicated to cover in a few slides. 

Instead, we'll just take advantage of the fact that the imaging pipeline is built on the tools of the scientific python ecosystem:

<center><img src="images/ehtim_dependency_graphic.png" alt="eht-imaging dependency graph" height=648 width=1152/></center>

Image credit: [Shaloo Shalini (@shaloo)](https://github.com/shaloo). For info on how this graphic was created, check out [shaloo's script](https://github.com/numpy/numpy.org/pull/23).

+++ {"slideshow": {"slide_type": "slide"}}

Let's run [the eht imaging pipeline](https://github.com/eventhorizontelescope/2019-D01-02) provided by the Event Horizons collaborators to help produce images from their [calibrated data](https://github.com/eventhorizontelescope/2019-D01-01). 

These repos with the helper-script for running the pipeline and the calibrated data have been included as submodules in `event_horizon_example/`.

```{code-cell} ipython3
%run event_horizons_example/2019-D01-02/eht-imaging/eht-imaging_pipeline.py  -i event_horizons_example/2019-D01-01/uvfits/SR1_M87_2017_101_lo_hops_netcal_StokesI.uvfits
```

+++ {"slideshow": {"slide_type": "subslide"}}

See? ...It's complicated. Here's the result:

```{code-cell} ipython3
---
slideshow:
  slide_type: subslide
---
fig = im_out.display(cfun=plt.cm.plasma)
```

+++ {"slideshow": {"slide_type": "slide"}}

## Parker Solar Probe

The [Parker Solar Probe](https://www.nasa.gov/content/goddard/parker-solar-probe-humanity-s-first-visit-to-a-star) was launched in 2018 to study the solar atmosphere, coming nearer to the sun than any previous space probe. Data from the Parker probe is already yielding [unexpected results](https://news.engin.umich.edu/2019/12/were-missing-something-fundamental-about-the-sun/?utm_source=newsletter&utm_medium=email&utm_campaign=January_2020).

One of the instruments on the probe is [SWEAP](http://sweap.cfa.harvard.edu/), a set of charged-particle detectors. Publicly available data from SWEAP can be found [here](http://sweap.cfa.harvard.edu/Data.html).

+++ {"slideshow": {"slide_type": "fragment"}}

Full disclosure: the original analysis for the [results published in Nature](https://www.nature.com/articles/s41586-019-1813-z) were produced with IDL, not Python. However, NASA just released [the first batch of data from the probe](https://sppgway.jhuapl.edu/) to the public, so let's see if we can't replicate some results...

+++ {"slideshow": {"slide_type": "skip"}}

### A quick aside: data formats

Data from the Parker probe is stored in NASA's [Common Data Format (CDF)](https://cdf.gsfc.nasa.gov/). Python libraries such as [spacepy](https://spacepy.github.io/) are used for I/O from the CDF format. As you might expect, `spacepy`'s `pycdf` module loads data from CDF files into NumPy arrays. Unfortunately, `spacepy`'s `pycdf` module depends on an external C-library, and there is not (yet) a `conda` recipe for installing it automatically.

+++ {"slideshow": {"slide_type": "skip"}}

To get around this I've used `spacepy.pycdf` to save a small amount of SWEAP in the more Python-friendly `.npz` format. I took data from the [SPC instrument collected on 11-08-18](http://sweap.cfa.harvard.edu/pub/data/sci/sweap/spc/L2/2018/11/). If you'd like to work with the full dataset, you can [manually install CDF](https://spacepy.github.io/install_linux.html#cdf), download the data (or any other dataset), and use `devlogs/parker_probe_velocity_log.py` as an example of how to load and interact with the raw data.

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
# See the full notebook to download this dataset
fname = 'data/parker_probe_spcL2_data_11-08-18.npz'
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
# The dataset is ~150MB, so it is not included in the repo.

import os, requests, tqdm
dsize = 152538956   # File size in bytes
dlink = 'https://www.dropbox.com/s/z45tbkqwjpyu6tz/parker_probe_spcL2_data_11-08-18.npz?dl=0'
if not os.path.exists(fname):
    r = requests.get(
        dlink,
        headers={'user-agent':'Wget/1.20 (linux-gnu)'},
        stream=True
    )
    with open(fname, 'wb') as fh:
        for chunk in tqdm.tqdm(r.iter_content(chunk_size=1024), total=dsize/1024):
            if chunk:
                fh.write(chunk)
```

+++ {"slideshow": {"slide_type": "slide"}}

Now that we have the data, let's try to replicate the top pane of [this image](http://sweap.cfa.harvard.edu/Images/example_spc_ql.png) from the [SWEAP data page](http://sweap.cfa.harvard.edu/Data.html).

We don't have time to discuss the data in detail, but the [Appendix 3 of the SWEAP Data User's Guide](http://sweap.cfa.harvard.edu/sweap_data_user_guide.pdf) outlines a procedure we can use to reproduce the desired figure. We start by loading the data:

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
# Load data from the SPC instrument on the Parker probe
data = np.load(fname)
# Measurement time (x-axis of image)
t = data['t']
# Edges of Voltage bins (y-axis of image)
mv_lo = data['mv_lo'].T
mv_hi = data['mv_hi'].T
# Differential charge flux density
dcfd = data['diff_charge_flux_density'].T
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
# The data include timestamps with microsecond resolution
# and 128 channels per data point
print(t.shape, mv_lo.shape, dcfd.shape)
t
```

```{code-cell} ipython3
---
slideshow:
  slide_type: subslide
---
# The CDF file uses a fill value (-1e31) to denote invalid data
print(dcfd)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
# Let's replace them so we can keep track of non-data in the arrays
for arr in (mv_lo, mv_hi, dcfd):
    arr[arr == -1e31] = np.nan
print(dcfd)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: subslide
---
# Upon closer inspection, only the first 30 of the 128 channels store valid data
np.sum(np.isfinite(dcfd), axis=1)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
# Limit computation to valid voltage bins
mv_lo, mv_hi, dcfd = mv_lo[:31,:], mv_hi[:31,:], dcfd[:31,:]
```

+++ {"slideshow": {"slide_type": "subslide"}}

After removing the unused data channels, there are still individual measurements that resulted in invalid data. Let's remove these as well.

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
# Mask out time samples that have bad data
bad_data = np.any(~np.isfinite(dcfd), axis=0)
t = t[~bad_data]
mv_lo = mv_lo[:, ~bad_data]
mv_hi = mv_hi[:, ~bad_data]
dcfd = dcfd[:, ~bad_data]
print("{} time samples out of {} discarded".format(bad_data.sum(), bad_data.shape[0]))
```

+++ {"slideshow": {"slide_type": "slide"}}

That takes care of the data munging, now on to the computation. The procedure in [Appendix 3 of the SWEAP data user's guide](http://sweap.cfa.harvard.edu/sweap_data_user_guide.pdf) boils down to a few straight-forward steps:

+++ {"slideshow": {"slide_type": "fragment"}}

#### 1. Compute the center of the voltage bins, $V$,  from `mv_lo` and `mv_hi`

+++ {"slideshow": {"slide_type": "fragment"}}

#### 2. Transform from particle *energy* to particle *velocity*

[Appendix 3](http://sweap.cfa.harvard.edu/sweap_data_user_guide.pdf) provides some helpful formulae:
 
$v_{p} = \sqrt{\frac{2qmv_{hi}}{m_{p}}} \frac{2}{\pi}E(\frac{mv_{lo}}{mv_{hi}})$
    
$dv_{p} = \sqrt{\frac{4qV}{m_{p}}} - v_{p}^{2}$

Where $v_{p}$ is the proton velocity, $dv_{p}$ is the bin width in velocity space, $q$ and $m_p$ are the charge and mass of the proton, respectively, and $E(x)$ represents an approximation to the elliptical integral.

+++ {"slideshow": {"slide_type": "subslide"}}

#### 3. Finally, compute the distribution of proton velocity, $F(v_{p})$, from the differential charge flux density measured by the instrument.

Again, [Appendix 3](http://sweap.cfa.harvard.edu/sweap_data_user_guide.pdf) gives us everything we need in describing the relationship between the differential charge flux density and the distribution of proton velocity:

$ dcfd = q v_{p} F(v_{p})dv_{p} \cdot 10^{8} $

```{code-cell} ipython3
---
slideshow:
  slide_type: subslide
---
# Step 1: Compute center and widths of voltage bins
V = (mv_hi + mv_lo) / 2
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
# We need the mass and charge of the proton for the next calculation
from scipy.constants import m_p as mp   # Proton mass [kg]
from scipy.constants import e as q      # Fundamental charge [C]
```

+++ {"slideshow": {"slide_type": "fragment"}}

We'll also need an approximation to the elliptic integral equation, $E(x)$. We'll use one of the approximations provided in `scipy.special`. 

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
from scipy.special import ellipe
```

```{code-cell} ipython3
---
slideshow:
  slide_type: subslide
---
# Step 2: Convert from energy -> velocity
v = np.sqrt(2 * q * mv_hi / mp) * (2 / np.pi) * ellipe(mv_lo/mv_hi)
dv = np.sqrt((4 * (q/mp) * V) - v**2)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
# Finally, compute the proton distribution as a function of proton velocity
Fv = (dcfd / (q* v* 10**8)) * (1 / dv)
```

+++ {"slideshow": {"slide_type": "slide"}}

Let's see how we did...

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
viz_kwargs = {
    "cmap" : plt.cm.plasma,
    "norm" : colors.LogNorm(vmin=1, vmax=200)
}
```

```{code-cell} ipython3
---
slideshow:
  slide_type: subslide
---
# Visualize
fig = plt.figure(figsize=(8,4)); ax = fig.add_subplot(111)
ax.pcolormesh(t[np.newaxis,:], v/1000, Fv, **viz_kwargs)
fig.colorbar(ax.collections[0])
fig.autofmt_xdate()
```

+++ {"slideshow": {"slide_type": "slide"}}

# Part III - Developing NumPy

+++ {"slideshow": {"slide_type": "slide"}}

# [Scope of NumPy](https://numpy.org/neps/scope.html)

The NumPy execution engine currently targets:

 * in-memory, homogenously-typed array data
 * cpu-based operations
 
Specialized hardware (e.g. GPUs), features for scalable computing (e.g. distributed arrays) are currently out of scope
 - Supporting libraries that provide these features *is in scope*: **extensibility** and **interoperability**

+++ {"slideshow": {"slide_type": "subslide"}}

Important guiding principles:
 - **Stability**: Foundational component of the scientific python ecosystem for going-on 15 years
 - **Interoperability**
   * NumPy is the standard array data structure within the scientific Python ecosystem
   * What about all the new array libraries?
     - [XArray](http://xarray.pydata.org/en/stable/)
     - [Dask Arrays](https://docs.dask.org/en/latest/array.html)
     - [Jax](https://jax.readthedocs.io/en/latest/)
     - [pydata sparse](https://sparse.pydata.org/en/latest/)
     - [PyTorch](https://pytorch.org/)
     - [TensorFlow](https://www.tensorflow.org/api_docs)

+++ {"slideshow": {"slide_type": "slide"}}

# How is NumPy Developed?

 - **Collaboratively** - https://github.com/numpy/numpy/

Commitment to stability means proposed changes must go through extensive design and review:
 - [Numpy Enhancement Proposals (NEPs)](https://numpy.org/neps/) - analogous to PEPs, specific to NumPy
   * Community-driven development and consensus among contributors/developers
     - Mailing list
     - PRs/Issues on GitHub
 - Steering council for high-level direction

+++ {"slideshow": {"slide_type": "slide"}}

# A case-study: `np.random`
 - Changes proposed in [NEP 19](https://numpy.org/neps/nep-0019-rng-policy.html), subsequently approved by the community via discussion on the mailing list and on GitHub.
 - Overhaul of `np.random` landed in version 1.17
 
   * Improve *performance* and *flexibility* without sacrificing stability

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
# Generate 1,000,000 random numbers the old way
old_rands = np.random.random(int(1e6))
print("Uniform random numbers from legacy np.random.random:\n  {}".format(old_rands))
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
# ... and the new way
from numpy.random import default_rng
rg = default_rng()
new_rands = rg.random(int(1e6))
print("Uniform random numbers with new tools:\n  {}".format(new_rands))
```

+++ {"slideshow": {"slide_type": "slide"}}

## Compatibility

There are many, many LOC (both in test suites and in production) that depend on the original `numpy.random`, so both the *interface* and the *results* must remain unchanged
 * <font color="green">**Upside: Stability**</font> - output of `np.random` remains consistent with previous versions
 * <font color="orange">**Downside: Discoverability**</font> - users need to know about new interface to access improvements

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
# Choose a seed for generator
seed = 1817

# Random numbers generated by np.random in v1.15
rands_from_v1_15 = np.load('data/npy_v1.15_random_seed1817_1000samples.npy')
# Generate random numbers with legacy interface
np.random.seed(seed)
legacy_rands = np.random.random(1000)

print("Arrays equivalent: ", np.allclose(rands_from_v1_15, legacy_rands))
```

+++ {"slideshow": {"slide_type": "skip"}}

It is possible (though clunky) to replicate legacy behavior with new interface

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
seed = 1817

from numpy.random import MT19937, RandomState, Generator
# Set random state with legacy seeding
rs = RandomState(seed)
mt = MT19937()
mt.state = rs.get_state()

# New interface for generation
rg = Generator(mt)
mt_rands = rg.random(1000)
print("Legacy: {}\nGenerator: {}".format(legacy_rands[:4], mt_rands[:4]))
print("Arrays equivalent: ", np.allclose(legacy_rands, mt_rands))
```

+++ {"slideshow": {"slide_type": "slide"}}

## Performance

The [PCG64](https://docs.scipy.org/doc/numpy/reference/random/bit_generators/pcg64.html) BitGenerator is a 
[significant improvement](http://www.pcg-random.org/) over the legacy Mersenne Twister in many areas, including speed:


```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
#NOTE: PCG64 is the new default bit_generator, so default_rng() equivalent to Generator(PCG64())
from numpy.random import default_rng
rg = default_rng()
num_samples = int(1e5) 

print("Uniform random numbers:")
%timeit np.random.random(num_samples)
%timeit rg.random(num_samples)
```

+++ {"slideshow": {"slide_type": "slide"}}

In addition, `Generator` includes improved methods for drawing samples from distributions.

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
print("Standard Normal:")
%timeit -n 100 np.random.standard_normal(num_samples)
%timeit -n 100 rg.standard_normal(num_samples) 
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
print("Standard Exponential:")
%timeit -n 100 np.random.standard_exponential(num_samples)
%timeit -n 100 rg.standard_exponential(num_samples)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
print("Standard Gamma:")
shape_param = 3.0
%timeit -n 100 np.random.standard_gamma(shape_param, num_samples)
%timeit -n 100 rg.standard_gamma(shape_param, num_samples)
```

+++ {"slideshow": {"slide_type": "skip"}}

## Parallel Generation

`np.random` includes new functionality to produce high-quality initital states for multiple generators to produce reproducible random numbers accross multiple processes.

For one example, let's take a look at `SeedSequence` and an [example from the documentation](https://numpy.org/devdocs/reference/random/multithreading.html)

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
Code(filename="mrng.py")
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
from mrng import MultithreadedRNG, default_rng
num_workers = 4
seed = 1817
n = int(1e7)

# Compare concurrent.futures multithreaded generation to single thread
rg = default_rng()
mg = MultithreadedRNG(n, seed=seed, threads=num_workers)

%timeit rg.standard_normal(n)
%timeit mg.fill()
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
# Maintain reproducible random number generators
ex1 = MultithreadedRNG(n, seed=seed, threads=num_workers)
ex2 = MultithreadedRNG(n, seed=seed, threads=num_workers)

# Generate numbers
ex1.fill()
ex2.fill()

# Results are reproducible
np.allclose(ex1.values, ex2.values)
```

+++ {"slideshow": {"slide_type": "slide"}}

# Part IV - Looking Ahead

+++ {"slideshow": {"slide_type": "slide"}}

# The changing landscape

 - In the early days, many new NumPy users were converts from languages like Matlab and IDL
   * See the [NumPy for Matlab users](https://docs.scipy.org/doc/numpy/user/numpy-for-matlab-users.html) article in the docs
   
 - **Now**: The scientific Python ecosystem (including libraries for data science and ML) is incredibly feature-rich and powerful, and is attracting many new users.
   * Users interested in specific applications (machine learning, image processing, geoscience, bioinformatics, etc.) end up interacting with NumPy indirectly

+++ {"slideshow": {"slide_type": "slide"}}

## Google Trends

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
# Data downloaded from google trends on 04-27-2020
!ls data/*.csv
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
!head data/datascience.csv
```

```{code-cell} ipython3
---
slideshow:
  slide_type: subslide
---
timeseries_dtype = np.dtype([
    ('date', 'datetime64[M]'),
    ('relpop', float)
])

parse_kwargs = {
    "skiprows" : 3,
    "delimiter" : ",",
    "dtype" : timeseries_dtype
}

fnames = ("numpy", "datascience", "matlab")

data = {
    fname : np.loadtxt("data/{}.csv".format(fname), **parse_kwargs) for fname in fnames
}
```

```{code-cell} ipython3
---
slideshow:
  slide_type: subslide
---
fig, ax = plt.subplots()
for name, vals in data.items():
    plt.plot(vals['date'], vals['relpop'], label=name)
ax.set_title('Google Trends (US): 2004 - Present')
ax.set_ylabel('Relative Popularity of Search Term [arb]')
fig.autofmt_xdate()
ax.legend();
```

```{code-cell} ipython3
---
slideshow:
  slide_type: subslide
---
def smooth(s, kernsize=21):
    s_padded = np.hstack((s[kernsize-1:0:-1], s, s[-2:-kernsize-1:-1]))
    kern = np.hamming(kernsize)
    res_padded = np.convolve(kern/kern.sum(), s_padded, mode='valid')
    # De-pad and renormalize
    return 100 * res_padded[kernsize//2:-kernsize//2+1] / res_padded.max()
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
fig, ax = plt.subplots()
for name, vals in data.items():
    plt.plot(vals['date'], smooth(vals['relpop']), label=name)
ax.set_title('Google Trends (US): 2004 - Present')
ax.set_ylabel('Relative Popularity of Search Term [arb]')
ax.legend();
```

+++ {"slideshow": {"slide_type": "slide"}}

# What's next for NumPy?

<center><img src="images/numpy_roadmap_graphic.png" alt="Numpy-near-future-graphic" height=648 width=1152/></center>

Image modified from [this PyData Amsterdam 2019 presentation](https://www.slideshare.net/RalfGommers/the-evolution-of-array-computing-in-python/14) by [Ralf Gommers](https://github.com/rgommers)

+++ {"slideshow": {"slide_type": "slide"}}

## Interoperability

Separate NumPy API from NumPy *execution engine*
 - Allow other libraries ([Dask](https://dask.org/), [CuPy](https://cupy.chainer.org/), [PyTorch](https://pytorch.org/), etc.) to support NumPy API.
 - Mitigate ecosystem fragmentation
   * E.g. don't want a re-implementation of `scipy` for each ML framework (`pytorch.scipy`, `tensorflow.scipy`, etc.)


+++ {"slideshow": {"slide_type": "slide"}}

### Current n-dimensional array landscape

<center><img src="images/array_landscape_now.png" alt="Array-ecosystem-now" height=648 width=1152/></center>

Images from this [talk at PyData NY 2019](https://www.slideshare.net/RalfGommers/pydata-nyc-whatsnew-numpyscipy-2019?next_slideshow=1) by [Ralf Gommers](https://github.com/rgommers)

+++ {"slideshow": {"slide_type": "slide"}}

### Vision for the future

<center><img src="images/array_landscape_vision.png" alt="array-ecosystem-vision" height=658 width=1152/></center>

Images from this [talk at PyData NY 2019](https://www.slideshare.net/RalfGommers/pydata-nyc-whatsnew-numpyscipy-2019?next_slideshow=1) by [Ralf Gommers](https://github.com/rgommers)

+++ {"slideshow": {"slide_type": "slide"}}

## One approach: `__array_function__` protocol

 - Proposed in [NEP 18](https://numpy.org/neps/nep-0018-array-function-protocol.html)
 - Array function protocol enabled by default as of version 1.17
 
<center><img src="images/array_function_descr.png" alt="array_function_protocol"/></center>
 
Image source: [this presentation](https://www.slideshare.net/RalfGommers/arrayfunction-conceptual-design-related-concepts?from_action=save) by [Ralf Gommers](https://github.com/rgommers)

+++ {"slideshow": {"slide_type": "slide"}}

### `__array_function__` example

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
import numpy as np

rg = np.random.default_rng()
x = rg.random((5000, 1000))

# Factorize with np.linalg
q, r = np.linalg.qr(x)
type(q), type(r)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
import dask.array as da

d = da.from_array(x, chunks=(1000, 1000))

# Same call signature!
q, r = np.linalg.qr(d)
type(q), type(r)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: skip
---
da.core.Array??
```

+++ {"slideshow": {"slide_type": "slide"}}

## Lessons learned from `__array_function__`

 - The `__array_function__` protocol has been successful, but has fallen short of universal adoption.
 - Valuable feedback from the community has resulted in [NEP 37](https://numpy.org/neps/nep-0037-array-module.html)
   * Defines `__array_module__` protocol
   * Currently under development (interested?)

+++ {"slideshow": {"slide_type": "slide"}}

## The problem with data types...

Current `dtype` system has some flexibility issues
 - Difficult to specify fully-featured types
 - Some mechanisms (e.g. casting rules) are difficult to extend to new types

+++ {"slideshow": {"slide_type": "subslide"}}

### Goal: Improve NumPy maintainability
 * Improve organization of dtype checking/comparison machinery
 * Use the same API for built-in and user-defined dtypes
 * Improve extensibility of API: facilitate future additions/modifications

+++ {"slideshow": {"slide_type": "fragment"}}

### User impact
 - Easier-to-use mechanism for defining fully-feature dtypes (including from Python)
 - Host of new dtypes for the ecosystem:
   * Physical units (cf. [astropy.unit](https://docs.astropy.org/en/stable/units/))
   * `bfloat16`, `int24`, etc.
   * Categorical types

+++ {"slideshow": {"slide_type": "fragment"}}

An approach to overhauling the dtype system is [currently being fleshed out in a new NEP](https://github.com/numpy/numpy/blob/a111b551ae940d7d5f8523fef1cf3589c6ba00a0/doc/neps/nep-0033-extensible-dtypes.rst).

+++ {"slideshow": {"slide_type": "slide"}}

## Improved SIMD incorporation for `ufuncs`

Strike a balance between **optimization** and **maintainability**

 - Define set of architecture-agnostic universal intrinsics
   * At build time, build code paths based on features available for the host architecture
   * At run time, detect which features are available and select which of available code paths to use
 - In the process of being formalized in a [draft NEP](https://github.com/mattip/numpy/blob/nep_simd/doc/neps/nep-XXXX-SIMD-optimizations.rst)
   * Preliminary work in support of this proposed enhancement can be found [here](https://github.com/numpy/numpy/pull/13421/files) and [here](https://github.com/numpy/numpy/pull/13516)
   
**N.B.** this approach (i.e. using universal intrinsics) was adopted by OpenCV

+++ {"slideshow": {"slide_type": "slide"}}

## Supporting language features: type annotations

Thinking about how best to support type annotations became especially important when they became an official core language feature in Python 3.7.

+++ {"slideshow": {"slide_type": "fragment"}}

This is the most cross-referenced issue in the NumPy GH repository (as of 04-27-2020):

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
Markdown(filename="data/top_issues_table.md")
```

+++ {"slideshow": {"slide_type": "subslide"}}

Work on type annotations is located in the [numpy-stubs](https://github.com/numpy/numpy-stubs) repository. Basic type annotations are supported:

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
Code(filename="type_annotations.py")
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
!mypy type_annotations.py
```

+++ {"slideshow": {"slide_type": "slide"}}

# ... and beyond: NumPy 2.0?

 - Major revision -> opportunity for refactoring/enhancements that break API
   * Weigh potential for improvements against the pain of breaking changes
   * Commitment to stability still a central theme!
 - So much new functionality being developed in external libraries
   * Changes that facilitate external development are priorities
 
A bit of the history surrounding the idea of NumPy 2.0 can be found [here](https://github.com/numpy/numpy/issues/9066)

+++ {"slideshow": {"slide_type": "slide"}}

# Getting involved

NumPy presents an opportunity to work on a project that is depended on by tens of millions of users (and counting). Here's how you can get involved:
 1. Where discussion happens:
  - [Numpy discussion mailing list](https://www.scipy.org/scipylib/mailing-lists.html)
  - Numpy community meetings - video conference every-other-week: [Community calendar link](https://calendar.google.com/calendar?cid=YmVya2VsZXkuZWR1X2lla2dwaWdtMjMyamJobGRzZmIyYzJqODFjQGdyb3VwLmNhbGVuZGFyLmdvb2dsZS5jb20)
  - slack channel: numpy-team.slack.com
 2. Contribute
   - [GitHub Issues](https://github.com/numpy/numpy/issues) and [open PRs](https://github.com/numpy/numpy/pulls) are a great entry point
     * If you want to get your hands dirty immediately, try starting with the [good first issue](https://github.com/numpy/numpy/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22) label
     * For challenges with a greater scope, try the [Enhancement](https://github.com/numpy/numpy/labels/01%20-%20Enhancement) or [Wish List](https://github.com/numpy/numpy/labels/23%20-%20Wish%20List) labels
   - Check out the discussion revolving around accepted and proposed [NEPs](https://numpy.org/neps/)

+++ {"slideshow": {"slide_type": "slide"}}

# Thank you!

If you have any questions, comments, or ideas please don't hesitate to contact me: rossbar@berkeley.edu

Also feel free to ask about/use/modify/contribute to this presentation on GitHub!

## Acknowledgements

> This project is funded in part by the Gordon and Betty Moore Foundation through
> [Grant GBMF5447](https://www.moore.org/grant-detail?grantId=GBMF5447f) and by
> the Alfred P. Sloan Foundation through 
> [Grant G-2017-9960](https://sloan.org/grant-detail/8222)
> to the University of California, Berkeley.

And a special thanks to [@stefanv](https://github.com/stefanv), [@seberg](https://github.com/seberg), and [@mattip](https://github.com/mattip/) for their generous input in the creation of this presentation.
