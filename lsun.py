"""
"""
import lmdb
import numpy
import scipy.misc
import StringIO


class Lsun(object):
    """
    """
    def __init__(self, path_lsun_dir):
        """
        """
        self._lmdb_path = path_lsun_dir
        self._lmdb_keys = []

        with lmdb.open(self._lmdb_path) as env:
            with env.begin(write=False) as txn:
                with txn.cursor() as cursor:
                    self._lmdb_keys = [k for k, v in cursor]

        self._key_indice = numpy.arange(len(self._lmdb_keys))
        self._key_position = 0

        numpy.random.shuffle(self._key_indice)

    def next_batch(self, batch_size):
        """
        """
        begin = self._key_position

        self._key_position += batch_size

        if self._key_position > len(self._key_indice):
            numpy.random.shuffle(self._key_indice)

            begin = 0

            self._key_position = batch_size

            assert batch_size <= len(self._key_indice)

        end = self._key_position

        images = []

        with lmdb.open(self._lmdb_path) as env:
            with env.begin(write=False) as txn:
                with txn.cursor() as cursor:
                    for i in xrange(begin, end):
                        val = cursor.get(self._lmdb_keys[self._key_indice[i]])

                        sio = StringIO.StringIO(val)

                        img = scipy.misc.imread(sio).astype(numpy.float32)

                        img /= 127.5
                        img -= 1.0

                        img = scipy.misc.imresize(img, 25)

                        images.append(img)

        return images
