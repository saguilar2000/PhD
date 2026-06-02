import struct

with open('setups/2fmri/eigenvectors.dat', 'rb') as f:
    while True:
        data = f.read(21 * 8)  # Read 21 doubles (168 bytes)
        if not data:
            break  # End of file
        zmax, dvnx_r, dvnx_i, dvny_r, dvny_i, dvnz_r, dvnz_i, dvix_r, dvix_i, dviy_r, dviy_i, dviz_r, dviz_i, dbx_r, dbx_i, dby_r, dby_i, dpn_r, dpn_i, dpi_r, dpi_i = struct.unpack('ddddddddddddddddddddd', data)
        print(f"zmax={zmax:.4f}, dvnx={dvnx_r:.4e} + {dvnx_i:.4e}i, dvny={dvny_r:.4e} + {dvny_i:.4e}i, dvnz={dvnz_r:.4e} + {dvnz_i:.4e}i, dvix={dvix_r:.4e} + {dvix_i:.4e}i, dviy={dviy_r:.4e} + {dviy_i:.4e}i, dviz={dviz_r:.4e} + {dviz_i:.4e}i, dbx={dbx_r:.4e} + {dbx_i:.4e}i, dby={dby_r:.4e} + {dby_i:.4e}i, dpn={dpn_r:.4e} + {dpn_i:.4e}i, dpi={dpi_r:.4e} + {dpi_i:.4e}i")