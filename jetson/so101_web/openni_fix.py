"""Make the python 'openni' package (written for OpenNI 2.2) safe with Orbbec's OpenNI 2.3 build.

Orbbec's OniDeviceInfo has an extra 256-byte serialNumber field (1028 bytes vs the wrapper's 772), so
oniDeviceGetInfo, which the wrapper calls on every Device open, writes past the buffer and corrupts the heap.
"""
import ctypes

from openni import _openni2 as c_api
from openni import openni2

ORBBEC_DEVINFO_SIZE = 4 * 256 + 4


def _get_device_info(self):
    if self._devinfo is None:
        info = c_api.OniDeviceInfo()
        ctypes.resize(info, max(ctypes.sizeof(info), 2 * ORBBEC_DEVINFO_SIZE))
        c_api.oniDeviceGetInfo(self._handle, ctypes.byref(info))
        self._devinfo = info
        self.uri = info.uri
    return self._devinfo


openni2.Device.get_device_info = _get_device_info
openni2.Device.device_info = property(_get_device_info)
