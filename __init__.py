# # Main function file - entry point into project execution
from .comparison_qrs_detectors import main

# Download libs for files from databases
from .mit_bih_download import get_data_anno, read_mit_bit_files
from .ludb_download import read_ludb_files
