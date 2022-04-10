import numpy as np
from PIL import Image
import os
from ISR.models import RDN


def doSR(destination, mode, target):
    img = Image.open(destination)
    lr_img = np.array(img)

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    if mode == 'x2':
        rdn = RDN(arch_params={'C': 6, 'D': 20, 'G': 64, 'G0': 64, 'x': 2})
        rdn.model.load_weights('models/rdn-C6-D20-G64-G064-x2.hdf5')
    else:
        rdn = RDN(arch_params={'C': 6, 'D': 20, 'G': 64, 'G0': 64, 'x': 4})
        rdn.model.load_weights('models/rdn-C6-D20-G64-G064-x4.hdf5')

    sr_img = rdn.predict(lr_img)
    out = Image.fromarray(sr_img)

    destination = "/".join([target, 'temp.png'])
    if os.path.isfile(destination):
        os.remove(destination)
    out.save(destination)
