import numpy as np
from PIL import Image          # o rasterio/gdal per i GeoTIFF
arr = np.array(Image.open("dtm_liguria_2017.tif"))
np.savetxt("dtm.txt", arr)