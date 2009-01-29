from PIL import Image
from numpy import array

def read(filename):
    """\
    Read the image data from filename and return an array object.
    """
    return array(Image.open(filename))

def write(filename, data):
    """\
    Save the image data (array object) to file.
    """
    Image.fromarray(data).save(filename)

