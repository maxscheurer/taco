import os
import glob


def fullfile(fn):
    thisdir = os.path.dirname(__file__)
    if os.path.isfile(os.path.join(thisdir, fn)):
        return os.path.join(thisdir, fn)
    elif os.path.isfile(fn):
        return fn
    else:
        return ""


# TODO: will be refactored
class TestdataCache():
    @property
    def files(self):
        thisdir = os.path.dirname(__file__)
        fnames = glob.glob(os.path.join(thisdir, "*.txt"))
        fnames += glob.glob(os.path.join(thisdir, "*.h5"))
        fnames += glob.glob(os.path.join(thisdir, "*.bas"))
        fnames += glob.glob(os.path.join(thisdir, "*.nwchem"))
        ret = {}
        for f in fnames:
            bname = os.path.splitext(os.path.basename(f))[0]
            ret[bname] = fullfile(f)
        return ret


cache = TestdataCache()
