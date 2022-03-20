from __future__ import print_function

import argparse
import os.path
import nibabel as nib
import numpy as np
import math

from six.moves import xrange
import pdb

"""
  getRandomPatchIndexs -- indices for random patches
  getOrderedPatchIndexes  -- indices for a regularly sampled grid
  getPatchesFromIndexes -- getPatchesFromIndexes
  getRandomPatches -- (RandomPatchIndexes -> PatchesFromIndexes)
  getOrderedPatches -- (OrderedPatchIndexes -> PatchesFromIndexes)
  imagePatchRecon -- Recon Image given patches and indices

  getbatch3d -- get batch of 3d patches
  getbatch   -- get batch of images -- if 3D call getbatch3d

"""
# ----------------------------------------------------------------------------
# ---------------------------- Saving Function ------------------------------
#
# Edge padding method for testing data
#
# ----------------------------------------------------------------------------

# Save np.array to nii
def save_nii(nii_image, nii_path):
    new_image = nib.Nifti1Image(nii_image, np.eye(4))  # np.array
    nib.save(new_image, nii_path)

# Load nii image into np.array
def load_nii(nii_path):
    img_nii = nib.load(nii_path)
    img_np = img_nii.get_data()
    return img_np


# ----------------------------------------------------------------------------
# ---------------------------- Padding Function ------------------------------
#
# Edge padding method for testing data
#
# ----------------------------------------------------------------------------

def limit_edge_patches(image_ori, target_shape):
    img_shape = image_ori.shape  # [51, 64, 64]
    pad = int((64 - img_shape[0]) / 2)  # floor_int, 6

    image_patches = np.zeros(target_shape, dtype=image_ori.dtype)  # [64, 64, 64]

    image_patches[0:pad,:,:] = np.repeat(np.expand_dims(image_ori[0, :, :], axis=0), pad, axis=0)
    image_patches[pad : img_shape[0] + pad,:,:] = image_ori
    image_patches[img_shape[0] + pad : 64, :,:] = np.repeat(np.expand_dims(image_ori[-1, :, :], axis=0), 64 - img_shape[0] - pad, axis=0)

    return image_patches




def edge_patches(image_ori, test_pad):
    image_shape = image_ori.shape  # [26,64,64]
    dim = len(image_shape)

    image_patch_size = ()
    for i in range(0, dim):
        image_patch_size += (image_shape[i] + 2 * test_pad[2-i],)  # [42,64,64]
    image_patches = np.zeros(image_patch_size, dtype=image_ori.dtype)

    image_patches[0:test_pad[2], :, :] = np.repeat(np.expand_dims(image_ori[0, :, :], axis=0), test_pad[2], axis=0)
    image_patches[test_pad[2]: image_shape[0] + test_pad[2], :, :] = image_ori
    image_patches[image_shape[0] + test_pad[2]: image_shape[0] + 2 * test_pad[2], :, :] = np.repeat(np.expand_dims(image_ori[-1, :, :], axis=0), test_pad[2], axis=0)

    return image_patches


# Reflect padding method for testing data
def reflect_patches(image_patches, target_shape):

    image_shape = image_patches.shape

    image_reflect = np.zeros(target_shape, dtype=image_patches.dtype)

    for i in range(target_shape[0]):  # 0 ~ 63
        if i < image_shape[0]:
            image_reflect[i,:,:] = image_patches[i,:,:]
        else:
            image_reflect[i,:,:] = image_patches[2 * image_shape[0] - 2 - i,:,:]

    return image_reflect




# -------------------------------------------------------------------------------
#
# Core Patch Code
#
# -------------------------------------------------------------------------------
# Get Indices
# -------------------------------------------------------------------------------


def get_random_patch_indexes(data, patch_size=None, num_patches=1, padding='VALID'):
    """Get data patch samples from a regularly sampled grid

    Create a list of indexes to access patches from the tensor of data, where each
    row of the index list is a list of array of indices.

    Returns:
      indexes: the index of the data denoting the top-left corner of the patch
    """
    if patch_size is None:
        patch_size = data.shape[:]

    dims = len(data.shape)
    indexes = np.zeros((num_patches,dims), dtype=np.int32)

    data_limits = list(data.shape[:])
    # for i in range(0,dims):
    #   data_limits += [data.shape[i]]

    if padding is 'VALID':
        for i in range(0, dims):
            data_limits[i] -= patch_size[i]

    for j in range(0, num_patches):
        for i in range(0, dims):
            indexes[j, i] = np.random.random_integers(0, data_limits[i])

    return indexes


def get_ordered_patch_indexes(data, patch_size=None, stride=None, padding='VALID'):
    """Get image patch samples from a regularly sampled grid

    Create the

    Returns:
      indexes: the index of the image denoting the top-left corner of the patch
    """

    dims = len(data.shape)

    if patch_size is None:
        internal_patch_size = data.shape[:]
    else:
        internal_patch_size = patch_size[:]
    for i in range(len(internal_patch_size), dims):
        internal_patch_size += [data.shape[i]]

    # Handle the stride
    if stride is None:
        stride = internal_patch_size[:]
    for i in range(len(stride), dims):
        stride += [data.shape[i]]

    total_patches = 1
    idx_all = []

    for i in range(0,dims):
      max_i = data.shape[i]
      if padding is 'VALID':
          max_i -= internal_patch_size[i]
      if max_i < 1:
          max_i = 1

      idx_all += [slice(1,max_i+1,stride[i])]

    grid = np.mgrid[idx_all]
    grid_size = grid.size
    indexes = np.transpose(grid.reshape(dims,int(grid_size/dims)))

    # Make sure to use 0 indexing
    indexes -= 1

    return indexes


# -------------------------------------------------------------------------------
# Get Patches
# -------------------------------------------------------------------------------


def get_patches_from_indexes(image, indexes, patch_size=None, padding='VALID', dtype=None):
    """Get image patches from specific positions in the image.

    Returns:
      patches: the image patches as a 4D numpy array
      indexes: the indexes of the image denoting the top-left corner of the patch in the image
               (just pass through really)
    """

    tmp_patch_size = list(image.shape[:])
    if patch_size is not None:
        # Ensure the patch size is of full data dimensions
        for i in range(0,min(len(image.shape), len(patch_size))):
            if patch_size[i] > 0:
                tmp_patch_size[i] = patch_size[i]

    if not(dtype):
        dtype = image.dtype

    dims = len(image.shape)
    num_patches = indexes.shape[0]

    patches_shape = (num_patches,)
    for i in range(0,dims):
        patches_shape += (tmp_patch_size[i],)
    patches = np.zeros(patches_shape, dtype=image.dtype)

    if padding is 'SAME':
        pad_slice = ()
        for i in range(0, dims):
            pad_slice += ((0, tmp_patch_size[i]),)
        image = np.pad(image, pad_slice, 'reflect')

    for i in range(0,num_patches):
        # Build the tuple of slicing indexes
        idx = ()
        for j in range(0,dims):
            idx += (slice(indexes[i,j],indexes[i,j]+tmp_patch_size[j]),)

        patches[i,...] = image[idx]

    return patches.astype(dtype)


def get_random_patches(image, patch_size=None, num_patches=1, padding='VALID', dtype=None):
    """Get image patch samples from a regularly sampled grid

    Create the

    Returns:
      patches: the image patches as a 4D numpy array
      indexes: the index of the image denoting the top-left corner of the patch
    """

    tmp_patch_size = list(image.shape[:])
    if patch_size is not None:
        # Ensure the patch size is of full data dimensions
        for i in range(0, min(len(image.shape), len(patch_size))):
            tmp_patch_size[i] = min(image.shape[i], patch_size[i])

    indexes = get_random_patch_indexes(image, tmp_patch_size, num_patches=num_patches, padding=padding)
    patches = get_patches_from_indexes(image, indexes, patch_size=tmp_patch_size, padding=padding, dtype=dtype)
    return [patches, indexes]


def get_ordered_patches(image, patch_size=None, stride=[1, 1, 1], num_patches=0, padding='VALID', dtype=None):
    """Get image patch samples from a regularly sampled grid

    Create the

    Returns:
      patches: the image patches as a 4D numpy array
      indexes: the index of the image denoting the top-left corner of the patch
    """

    tmp_patch_size = list(image.shape[:])
    if patch_size is not None:
        # Ensure the patch size is of full data dimensions
        for i in range(0,min(len(image.shape), len(patch_size))):
            tmp_patch_size[i] = min(image.shape[i], patch_size[i])

    indexes = get_ordered_patch_indexes(image, tmp_patch_size, stride=stride, padding=padding)

    total_patches = indexes.shape[0]
    if num_patches > total_patches:
        num_patches = total_patches

    if num_patches > 0:
        indexes = indexes[0:num_patches,...]

    patches = get_patches_from_indexes(image, indexes, patch_size=tmp_patch_size, padding=padding, dtype=dtype)

    return [patches, indexes]


# -------------------------------------------------------------------------------
#
# Image Patch Reconstruction
#
# -------------------------------------------------------------------------------
def image_patch_smooth_recon(output_size, patches, indexes, dtype=None, sigma=0.0):
    if not(dtype):
      dtype = patches.dtype

    dims = len(output_size)
    patch_size = []
    for i in range(1, dims+1):
      patch_size += [patches.shape[i]]

    # Check that the patches match the output shape, squeeze if necessary
    if len(patches.shape)-1 > dims:
      patches = np.squeeze(patches,axis=len(patches.shape)-1)

    padded_size = ()
    for i in range(0,dims):
      padded_size += (output_size[i]+patch_size[i],)
    padded_image = np.zeros(padded_size, dtype=np.float32)
    sum_image = np.zeros(padded_image.shape, dtype=np.float32)

    # Setup the weight mask
    weight_mask=np.zeros(patch_size,dtype=np.float32)
    mask_slice = ()
    for i in range(0,dims):
      half_i = 0.5*(patch_size[i]-1)
      mask_slice += (slice(-half_i,half_i+1),)
    mask_grid = np.mgrid[mask_slice]

    for i in range(0,dims):
      sigma_i = sigma*patch_size[i]
      scalar = 1.0/(sigma_i*sigma_i)
      weight_mask += scalar*pow(mask_grid[i],2.0)
    weight_mask = np.exp(-0.5*weight_mask)
    weight_mask[weight_mask<1e-8] = 0

    for i in xrange(0,patches.shape[0]):
      # Build the tuple of slicing indexes
      idx = ()
      for j in range(0,dims):
        idx += (slice(indexes[i,j],indexes[i,j]+patch_size[j]),)

      padded_image[idx] += np.multiply(patches[i,...],weight_mask)
      sum_image[idx] += weight_mask

    # Make sure the denominator is good
    sum_image[sum_image<1e-8] = 1
    image = np.true_divide(padded_image, sum_image)

    # Prepare the output
    output_idx = ()
    for i in range(0,dims):
      output_idx += (slice(0,output_size[i]),)
    output = image[output_idx]
    return output.astype(dtype)


def image_patch_recon(output_size, patches, indexes, dtype=None, sigma=0.0):
    if not(dtype):
      dtype = patches.dtype

    if sigma>0:
       return image_patch_smooth_recon(output_size,patches,indexes,dtype,sigma)

    dims = len(output_size)
    patch_size = []
    for i in range(1,dims+1):
      patch_size += [patches.shape[i]]

    # Check that the patches match the output shape, squeeze if necessary
    # if len(patches) > dims:
    #   patches = np.squeeze(patches,axis=len(patches.shape)-1)

    padded_size = ()
    for i in range(0,dims):
      padded_size += (output_size[i]+patch_size[i],)
    padded_image = np.zeros(padded_size, dtype=patches.dtype)
    sum_image = np.zeros(padded_image.shape, dtype=np.float32)

    for i in xrange(0,patches.shape[0]):
      # Build the tuple of slicing indexes
      idx = ()
      for j in range(0,dims):
        idx += (slice(indexes[i,j],indexes[i,j]+patch_size[j]),)

      padded_image[idx] += patches[i,...]
      sum_image[idx] += 1

    # Make sure the denominator is good
    sum_image[sum_image<1] = 1
    image = np.true_divide(padded_image, sum_image);

    # Prepare the output
    output_idx = ()
    for i in range(0,dims):
      output_idx += (slice(0,output_size[i]),)
    output = image[output_idx]
    return output.astype(dtype)


def image_patch_smooth_recon_one_hot(output_size, patches, indexes, num_classes, dtype=None,threed=False,sigma=0.0):

  if not(dtype):
    dtype = patches.dtype

  dims = len(output_size)-1
  patch_size = [1]*dims
  for i in range(0,len(patches.shape)-1):
    patch_size[i] = patches.shape[i+1]

  # Check that the patches match the output shape, squeeze if necessary
  if len(patches.shape) > dims+1:
    patches = np.squeeze(patches,axis=-1)
  if len(patches.shape)-1 < dims:
    patches = np.expand_dims(patches,axis=-1)

  # Get the max indexes
  max_index = [0]*dims
  for i in range(0,len(indexes)):
    for j in range(0,dims):
      if max_index[j] < indexes[i,j]:
        max_index[j] = indexes[i,j]

  padded_size = ()
  for i in range(0,dims):
    padded_size += (max_index[i]+patch_size[i],)
  padded_size += (output_size[-1],)
  padded_image = np.zeros(padded_size, dtype=np.float32)
  sum_image = np.zeros(padded_image.shape, dtype=np.float32)

  # Setup the weight mask
  weight_mask=np.zeros(patch_size,dtype=np.float32)
  mask_slice = ()
  for i in range(0,dims):
    half_i = 0.5*(patch_size[i]-1)
    mask_slice += (slice(-half_i,half_i+1),)
  mask_grid = np.mgrid[mask_slice]

  for i in range(0,dims):
    sigma_i = sigma*patch_size[i]
    scalar = 1.0/(sigma_i*sigma_i)
    weight_mask += scalar*pow(mask_grid[i],2.0)
  weight_mask = np.exp(-0.5*weight_mask)
  weight_mask[weight_mask<1e-8] = 0
  weight_mask = np.repeat(np.expand_dims(weight_mask,len(weight_mask)+1),num_classes,axis=dims)

  for i in xrange(0,patches.shape[0]):
    # Build the tuple of slicing indexes
    idx = ()
    for j in range(0,dims):
      idx += (slice(indexes[i,j],indexes[i,j]+patch_size[j]),)
    idx += (slice(0,num_classes),)

    p_i = patches[i,...].astype(int)
    hot_i = np.eye(num_classes)[p_i]
    padded_image[idx] += np.multiply(hot_i,weight_mask)
    sum_image[idx] += weight_mask

  # Make sure the denominator is good
  sum_image[sum_image<1e-8] = 1
  image = np.true_divide(padded_image, sum_image)

  # Prepare the output 
  output_idx = ()
  for i in range(0,dims):
    output_idx += (slice(0,output_size[i]),)
  output = image[output_idx]
  return output.astype(np.float32)


def image_patch_recon_one_hot(output_size, patches, indexes, num_classes, dtype=None,threed=False,sigma=0.0):

  if sigma>0:
     return image_patch_smooth_recon_one_hot(output_size,patches,indexes,num_classes,dtype,threed,sigma)

  if not(dtype):
    dtype = patches.dtype

  # Need to subtract one dim for one-hot encoding
  dims = len(output_size)-1
  patch_size = [1]*dims
  for i in range(0,len(patches.shape)-1):
    patch_size[i] = patches.shape[i+1]

  # Check that the patches match the output shape, squeeze if necessary
  if len(patches.shape) > dims+1:
    patches = np.squeeze(patches,axis=-1)
  if len(patches.shape)-1 < dims:
    patches = np.expand_dims(patches,axis=-1)

  # Get the max indexes
  max_index = [0]*dims
  for i in range(0,len(indexes)):
    for j in range(0,dims):
      if max_index[j] < indexes[i,j]:
        max_index[j] = indexes[i,j]

  padded_size = ()
  for i in range(0,dims):
    padded_size += (max_index[i]+patch_size[i],)
  # Add another one-hot dim back on the end
  padded_size += (output_size[-1],)
  padded_image = np.zeros(padded_size, dtype=patches.dtype)
  sum_image = np.zeros(padded_image.shape, dtype=np.float32)

  for i in xrange(0,patches.shape[0]):
    # Build the tuple of slicing indexes
    idx = ()
    for j in range(0,dims):
      idx += (slice(indexes[i,j],indexes[i,j]+patch_size[j]),)
    idx += (slice(0,num_classes),)

    p_i = patches[i,...].astype(int)
    hot_i = np.eye(num_classes, dtype=int)[p_i]
    padded_image[idx] += hot_i
    sum_image[idx] += 1

  # Make sure the denominator is good
  sum_image[sum_image<1] = 1
  image = np.true_divide(padded_image, sum_image);

  # Prepare the output 
  output_idx = ()
  for i in range(0,dims):
    output_idx += (slice(0,output_size[i]),)
  output = image[output_idx]
  return output.astype(np.float32)


# -------------------------------------------------------------------------------
# Crop Image
# -------------------------------------------------------------------------------
def crop_image(image, offset=None):

  dims = len(image.shape)
  offset_size = []
  for i in range(0,dims):
    offset_size += [0]
  if offset is not None:
    for i in range(0,min(dims,len(offset))):
      offset_size[i] = offset[i]

  crop_slice = ()
  for i in range(0,dims):
    crop_slice += (slice(offset_size[i],image.shape[i]-offset_size[i]),)

  return image[crop_slice]

# -----------------------------------------------------------------------------------
#
# Main Function
#
# -----------------------------------------------------------------------------------
if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Load an image for patch sampling.')
  parser.add_argument('input', nargs=1, help='NIfTI image input file.')
  parser.add_argument('output', nargs=1, help='NIfTI image patch output file.')
  parser.add_argument('-n','--num_samples', type=int, help='number of image patch samples to extract', default=0)
  parser.add_argument('-r','--random', help='Perform random patch sampling from the image', action='store_true')
  parser.add_argument('-p','--patch_size', type=int, nargs='+', help='Set the patch size in voxels', default=[1])
  parser.add_argument('-s','--stride', type=int, nargs='+', help='Set the patch stride in voxels', default=[1])
  parser.add_argument('--sigma', type=float, help='Reconstruction weight mask smoothing parameter, default=0.0', default=0.0)
  parser.add_argument('--recon', help='File name for to create a reconstructed image from the sampled patches')
  args = parser.parse_args()


  if not os.path.isfile(args.input[0]):
    raise ValueError('Failed to find the file: ' + f)
  print('Loading file: %s' % args.input[0])
  nifti_image = nib.load(args.input[0])
  image = nifti_image.get_data()
  print('Loaded image with data of size: '+str(image.shape))

  # Get the data dimensionality
  dims = len(image.shape)

  patch_size = []
  for i in range(0,dims):
    patch_size += [1]
  # Set the patch size from the input 
  for i in range(0,min(dims,len(args.patch_size))):
    patch_size[i] = min(image.shape[i],args.patch_size[i])

  print('Patch size = %s' %(patch_size,))
  print('Random sampling = %r' % args.random)

  if args.random:
    [patches, indexes] = getRandomPatches(image, patch_size, num_patches=args.num_samples, padding='SAME')
  else:
    stride = []
    for i in range(0,dims):
      stride += [1]
    for i in range(0,min(dims,len(args.stride))):
      stride[i] = max(1,args.stride[i])
    print('Stride: '+str(stride))

    [patches, indexes] = getOrderedPatches(image, patch_size, stride=stride, padding='VALID', num_patches=args.num_samples)

  print('Patch sampling complete.')
  print('Got %d patches from the image...' % patches.shape[0])


  out_patches = np.zeros(patch_size + [patches.shape[0]], dtype=image.dtype)
  for i in range(0,patches.shape[0]):
    out_patches[..., i] = patches[i, ...]

  print('Saving the patch image out: %s' % args.output[0])
  output = nib.Nifti1Image(out_patches, nifti_image.affine)
  nib.save(output, args.output[0])


  if args.recon:
    print('Saving reconstructed image out: %s' % args.recon)
    r_image = imagePatchRecon(image.shape, patches, indexes, sigma=args.sigma)
    output = nib.Nifti1Image(r_image, nifti_image.affine, header=nifti_image.header)
    nib.save(output, args.recon)


