

def vcc_mat_to_txt(mat_file,
                   output_txt,
                   normalize="sum",
                   subtract_bg=True,
                   crop=True):
    """
    Convert ProfMon VCC .mat file to distgen/IMPACT xy_dist text file.
    Output units are microns [um].
    """

    # ----------------------------
    # Load MAT file (v7.3 HDF5)
    # ----------------------------
    with h5py.File(mat_file, 'r') as f:
        data = f['data']
        img  = np.array(data['img'])
        ncol = int(np.array(data['nCol']).squeeze())
        nrow = int(np.array(data['nRow']).squeeze())
        res  = float(np.array(data['res']).squeeze())  # µm per pixel

    # Fix transpose if needed
    if img.shape != (nrow, ncol):
        img = img.T

    print(f"Resolution from file: {res} um/px")

    # ----------------------------
    # Background subtraction
    # ----------------------------
    if subtract_bg:
        img = img - np.percentile(img, 5)
        img[img < 0] = 0

    # ----------------------------
    # Optional crop
    # ----------------------------
    if crop:
        mask = img > img.max() * 0.02
        rows = np.where(mask.sum(axis=1) > 0)[0]
        cols = np.where(mask.sum(axis=0) > 0)[0]

        if len(rows) > 0 and len(cols) > 0:
            img = img[rows.min():rows.max()+1,
                      cols.min():cols.max()+1]

    # ----------------------------
    # Normalize
    # ----------------------------
    if normalize == "sum":
        img = img / img.sum()
    elif normalize == "max":
        img = img / img.max()
    else:
        raise ValueError("normalize must be 'sum' or 'max'")

    # ----------------------------
    # Write distgen format
    # ----------------------------
    ny, nx = img.shape
    width_x = nx * res      # µm
    width_y = ny * res      # µm

    header = f"""x {width_x} 0 [um]
y {width_y} 0 [um]"""

    np.savetxt(output_txt,
               np.flipud(img),
               header=header,
               comments='')

    print(f"Saved: {output_txt}")
    print(f"Final shape: {img.shape}")
    print(f"Physical size: {width_x:.1f} um x {width_y:.1f} um")

    return img


def run_fudge_optimization(tao):
    """
    Runs Tao energy optimization using the enabled fudge variables.
    """
    tao.cmd('veto dat *')
    tao.cmd('veto var *')
    tao.cmd('use dat BC1.energy[1]')
    tao.cmd('use var linac_fudge[1]')
    tao.cmd('run')
   
    tao.cmd('veto dat *')
    tao.cmd('veto var *')
    tao.cmd('use dat BC2.energy[1]')
    tao.cmd('use var linac_fudge[2]')
    tao.cmd('run')
    
    tao.cmd('veto dat *')
    tao.cmd('veto var *')
    tao.cmd('use dat L3.energy[2]')
    tao.cmd('use var linac_fudge[3]')
    tao.cmd('run')


# In[10]:


def configure_energy_target(tao, energy_data, erro_p: dict):
    """
    Configure Tao energy matching targets for L1, L2, and L3 sections.

    Parameters:
        tao (pytao.Tao): Active Tao session.
        energy_data (dict): Dict with measured energies in GeV, e.g.
                            {'L1': 0.135, 'L2': 0.25, 'L3': 5.0}
    """
    # Map section name → (tao data name, variable name)
    config_map = {
        'L1': ('BC1.energy[1]', 'linac_fudge[1]'),
        'L2': ('BC2.energy[1]', 'linac_fudge[2]'),
        'L3': ('L3.energy[2]',  'linac_fudge[3]')
    }

    for section, (data_name, fudge_var) in config_map.items():
        if section not in energy_data:
            print(f"[Warning] {section} energy not in input, skipping.")
            continue

        meas_value = energy_data[section]*erro_p
        weight = 1  # You can adjust this if needed

        # --- ensure the datum exists ---
        try:
            tao.cmd(f"set data -silent {data_name}|merit_type = target")
        except RuntimeError:
            print(f"Creating data entry for {data_name}")
            tao.data_d2_create(
                d2_name=data_name.split('.')[0],
                n_d1_data=1,
                d_data_arrays_name_min_max='energy^^1^^1'
            )

        # --- configure target ---
        tao.cmd(f"set data -silent {data_name}|data_source = lat")
        tao.cmd(f"set data -silent {data_name}|weight = {weight}")
        tao.cmd(f"set data -silent {data_name}|meas = {meas_value}")
        tao.cmd(f"set data -silent {data_name}|merit_type = target")

        # --- enable fudge variable ---
        tao.cmd('veto var *')
        tao.cmd(f'use var {fudge_var}')

        print(f"Configured {section}: {data_name} with meas={meas_value:.4f} GeV")
