import struct
def unpack(stream, fmt):
    size = struct.calcsize(fmt)
    buffer_ = stream.read(size)
    return struct.unpack(fmt,buffer_)

def mat(f):
    dims = unpack(f, 'ii')
    x = []
    while True:
        try:
            x.append(unpack(f,'d')[0])
        except:
            break
    return x
