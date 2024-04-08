print((xspace, yspace, zspace))
strip.cell_data["Bx"] = B_sx.flatten("F")
strip.cell_data["By"] = B_sy.flatten("F")
strip.cell_data["Bz"] = B_sz.flatten("F")
strip.cell_data["B_mag"] = np.sqrt(
    B_sx.flatten("F") ** 2 + B_sy.flatten("F") ** 2 + B_sz.flatten("F") ** 2
)
