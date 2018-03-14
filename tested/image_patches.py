import unittest
import tensorflow as tf
import numpy as np


def split_patches_into_batch(images, patch_size=[2, 2], channels=1, reversible=False):
    kernel = [1] + patch_size + [1]
    stride = kernel if reversible else [1, 1, 1, 1]
    result = tf.extract_image_patches(images, kernel, strides=stride, rates=[1, 1, 1, 1], padding='VALID')
    return tf.reshape(result, [-1, np.prod(patch_size) * channels])


def op_join_patches_from_batch(patches, original_size, patch_size=[2, 2], channels=1):

    def implementation(patches_numpy):
        patch_fragments = np.reshape(patches_numpy, [-1, patch_size[1] * channels])
        patch_count = np.divide(original_size, patch_size).astype(int).tolist()
        batch_count = int(np.prod(patch_fragments.shape) / (np.prod(original_size) * channels))
        print(patch_fragments)
        return np.array([
            patch_fragments[i_hor + rows_in_patch * patch_count[1] + i_ver * patch_size[0] * patch_count[1]]
            for i_ver in range(patch_count[0]*batch_count)
            # Secondly move to the next
            for i_hor in range(patch_count[1])
            # First output all in the current row of the current patch
            for rows_in_patch in range(patch_size[0])
        ])

    result = tf.py_func(implementation, [patches], patches.dtype)
    return tf.reshape(result, shape=[-1] + original_size + [channels])


# Test

def _get_index_shape(shape):
    return tf.reshape(list(range(np.prod(shape))), shape)


class _Tests(tf.test.TestCase):

    def _testIfPatchIsReversible(self, patch_count, patch_size, channels):
        original_size = np.multiply(patch_count, patch_size).tolist()
        indexes = tf.tile(_get_index_shape([1] + original_size + [1]), [1, 1, 1, channels])
        patches = split_patches_into_batch(indexes, reversible=True, patch_size=patch_size, channels=channels)
        joined = op_join_patches_from_batch(patches, original_size, patch_size=patch_size, channels=channels)

        with tf.Session() as session:
            joined.eval()
            #self.assertAllEqual(tf.shape(patches).eval(), [np.prod(patch_count), np.prod(patch_size) * channels])
            #self.assertAllEqual(indexes.eval(), joined.eval())

    def testPatches(self):
        # self._testIfPatchIsReversible(patch_count=[2, 2], patch_size=[2, 2], channels=1)
        # self._testIfPatchIsReversible(patch_count=[2, 2], patch_size=[2, 2], channels=2)
        # self._testIfPatchIsReversible(patch_count=[2, 2], patch_size=[2, 2], channels=3)
        # self._testIfPatchIsReversible(patch_count=[1, 2], patch_size=[2, 2], channels=2)
        # self._testIfPatchIsReversible(patch_count=[5, 5], patch_size=[1, 2], channels=2)
        # self._testIfPatchIsReversible(patch_count=[5, 5], patch_size=[1, 2], channels=2)
        # self._testIfPatchIsReversible(patch_count=[5, 5], patch_size=[1, 5], channels=2)
        self._testIfPatchIsReversible(patch_count=[3, 3], patch_size=[2, 2], channels=1)


if __name__ == '__main__':
    unittest.main()
