import os
import glob

from taco.testdata.cache import fullfile


# TODO: will be refactored
class DataCache():
    @property
    def files(self):
        thisdir = os.path.dirname(__file__)
        fnames = glob.glob(os.path.join(thisdir, "*.txt"))
        ret = {}
        for f in fnames:
            bname = os.path.splitext(os.path.basename(f))[0]
            ret[bname] = fullfile(f)
        return ret

    @property
    def jfiles(self):
        thisdir = os.path.dirname(__file__)
        fnames = glob.glob(os.path.join(thisdir, "*.json"))
        ret = {}
        for f in fnames:
            bname = os.path.splitext(os.path.basename(f))[0]
            ret[bname] = fullfile(f)
        return ret


data = DataCache()
