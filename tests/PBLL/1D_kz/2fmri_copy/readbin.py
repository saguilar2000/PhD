import struct

with open('eigenvectors.dat', 'rb') as f:
    while True:
        data = f.read(7 * 8)  # Read 7 doubles (56 bytes)
        if not data:
            break  # End of file
        zmax, dvnx, dvny, dvix, dviy, dbx, dby = struct.unpack('ddddddd', data)
        print(f"zmax={zmax:.4f}, dvnx={dvnx:.4e}, dvny={dvny:.4e}, dvix={dvix:.4e}, dviy={dviy:.4e}, dbx={dbx:.4e}, dby={dby:.4e}")