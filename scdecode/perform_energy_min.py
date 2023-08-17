
import glob
from prep_pdb import minimize_energy

clean_files = glob.glob('clean_pdbs/*.pdb')

for f in clean_files:
    try:
        minimize_energy(f, out_fmt_str='energy_min_pdbs/%s.pdb')
    except Exception as e:
        print('On file %s, failed with exception:\n%s'%(f, str(e)))

