import os
import sys
import argparse
from time import time

def line_to_coor(line, form):
    if form == 'nucleic_atom':
        st = line.split()
        name = line[12:17].strip().strip('\'')
        x = float(line[30:38]) 
        y = float(line[38:46]) 
        z = float(line[46:54]) 
        atom = st[-1]
        idx = int(line[22:26])
        return name, x, y, z, atom, idx
    elif form == 'ligand_mol2':
        st = line.split()
        name = st[1].strip('\'')
        x = float(line[16:26])
        y = float(line[26:36])
        z = float(line[36:46])
        atom = st[5]
        atom = atom.split('.')[0]
    if len(name) > 3:
        name = name[:3]
    while not atom[0].isalpha():
        atom = atom[1:]
    while not atom[-1].isalpha():
        atom = atom[:-1]
    return name, x, y, z, atom

def _set_num(x, l):
	return ' ' * (l - len(str(x))) + str(x)

def _set_coord(x, l):
	xx = str(round(x, 4))
	return ' ' * (l - len(xx)) + xx

def get_refined_pose_file(mol2_file, output_mol2_file, ligand):
	lines = []
	flag = 0
	a2a = {}
	atom_id = 1
	nonH_atom = 1
	bond_id = 1
	with open(mol2_file, 'r') as f:
		for line in f:
			if (line[:13] == '@<TRIPOS>ATOM'):
				flag = 1
				lines.append(line)
				continue
			if (line[:13] == '@<TRIPOS>BOND'):
				flag = 2
				lines.append(line)
				assert len(ligand) == nonH_atom - 1
				continue
			if (line[:13] == '@<TRIPOS>SUBS'):
				flag = 0
				lines.append(line)
				continue
			if flag == 1:
				name, x, y, z, atom = line_to_coor(line, 'ligand_mol2')
				if (atom != 'H'):
					a2a[atom_id] = nonH_atom
					x = ligand[nonH_atom - 1][0]
					y = ligand[nonH_atom - 1][1]
					z = ligand[nonH_atom - 1][2]
					st = _set_num(nonH_atom, 7) + line[7:16] + _set_coord(x, 10) + _set_coord(y, 10) + _set_coord(z, 10) + line[46:]
					lines.append(st)
					nonH_atom += 1
				atom_id += 1
				continue
			if flag == 2:
				st = line.split()
				x = int(st[1])
				y = int(st[2])
				if x in a2a and y in a2a:
					x = a2a[x]
					y = a2a[y]
					l = _set_num(bond_id, 6) + _set_num(x, 5) + _set_num(y, 5)
					split_parts = line[16:].split()
					if len(split_parts) >= 2:
				            bond_type = split_parts[-1]
					else:
					    bond_type = line[16:].strip()
					l += " " + bond_type + "\n"
					bond_id += 1
					lines.append(l)
				continue
			lines.append(line)
	for i in range(len(lines)):
		if '@<TRIPOS>MOLECULE' in lines[i]:
			l = i
			break
	original_fields = lines[l + 2].split()
	new_part = _set_num(nonH_atom - 1, 5) + _set_num(bond_id - 1, 6)
	if len(original_fields) >= 3:
		remaining = " " + " ".join(original_fields[2:]) 
	else:
		remaining = ""
	lines[l + 2] = new_part + remaining + '\n'

	with open('tmp_mol.mol2', 'w') as f:
		for line in lines:
			f.write(line)

	os.system('obminimize tmp_mol.mol2 > tmp_mol.pdb')
	os.system(f'obabel -ipdb tmp_mol.pdb -omol2 -O {output_mol2_file}')
	os.system('rm tmp_mol.mol2')
	os.system('rm tmp_mol.pdb')
	#os.system(f'mv tmp_mol.mol2 {output_mol2_file}')





