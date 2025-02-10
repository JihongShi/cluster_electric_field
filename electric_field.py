import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

class WaterShellAnalysis:
    def __init__(self, trajectory_file, shell_ranges=None, output_dir='output.dir'):
        self.trajectory_file = trajectory_file
        self.k = 8.99e9  
        self.e = 1.602e-19
        self.charges = {'Fe': 3.0, 'O': -0.82, 'H': 0.41} 
        self.shell_ranges = shell_ranges or [(0, 3), (3, 5), (5, 10)] 
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.atom_type_map = {1: 'O', 2: 'H', 3: 'Fe'} 
        
    def read_lammps_trajectory(self):
        """Read LAMMPS trajectory file and return frames."""
        frames = []
        current_frame = None
        
        with open(self.trajectory_file, 'r') as f:
            while True:
                line = f.readline()
                if not line:
                    break
                    
                if line.startswith('ITEM: TIMESTEP'):
                    if current_frame is not None:
                        frames.append(current_frame)
                    current_frame = {'positions': [], 'types': [], 'box': None}
                    f.readline()
                    
                elif line.startswith('ITEM: NUMBER OF ATOMS'):
                    num_atoms = int(f.readline())
                    
                elif line.startswith('ITEM: BOX BOUNDS'):
                    box = []
                    for _ in range(3):
                        lo, hi = map(float, f.readline().split())
                        box.append([lo, hi])
                    current_frame['box'] = np.array(box)
                    
                elif line.startswith('ITEM: ATOMS'):
                    for _ in range(num_atoms):
                        atom_data = f.readline().split()
                        atom_type = int(atom_data[1])
                        # Skip counter ions (type 4)
                        if atom_type != 4:
                            current_frame['types'].append(atom_type)
                            current_frame['positions'].append([float(atom_data[2]), 
                                                            float(atom_data[3]), 
                                                            float(atom_data[4])])
                            
        if current_frame is not None:
            frames.append(current_frame)
            
        return frames

    def get_water_indices(self, types, o_idx):
        """Get indices of a water molecule given the oxygen index."""
        if o_idx + 2 >= len(types) or types[o_idx] != 1 or \
           types[o_idx + 1] != 2 or types[o_idx + 2] != 2:
            return []
        return [o_idx, o_idx + 1, o_idx + 2]

    def calculate_electric_field(self, positions, types, target_idx, exclude_indices=None):
        """Calculate the electric field at the target atom from all other atoms."""
        electric_field = {}
        target_pos = np.array(positions[target_idx])
        exclude_indices = set(exclude_indices or [])
        
        for i, (pos, type_num) in enumerate(zip(positions, types)):
            if i == target_idx or i in exclude_indices:
                continue
                
            pos = np.array(pos)
            r_vector = target_pos - pos
            r_magnitude = np.linalg.norm(r_vector)
            if r_magnitude < 1e-10:
                continue
                
            r_meter = r_magnitude * 1e-10
            symbol = self.atom_type_map[type_num]
            source_charge = abs(self.charges[symbol]) * self.e
            E_magnitude = (self.k * source_charge) / (r_meter * r_meter)
            E_magnitude_angstrom = E_magnitude * 1e-10
            
            if symbol in ['Fe', 'H']:
                E_vector = E_magnitude_angstrom * (r_vector / r_magnitude)
            else:
                E_vector = -E_magnitude_angstrom * (r_vector / r_magnitude)
                
            electric_field[i] = E_vector
                
        return electric_field

    def calculate_electric_force(self, positions, types, target_idx, exclude_indices=None):
        """Calculate the electric force on the target atom from all other atoms."""
        electric_force = {}
        target_pos = np.array(positions[target_idx])
        target_symbol = self.atom_type_map[types[target_idx]]
        exclude_indices = set(exclude_indices or [])
        
        for i, (pos, type_num) in enumerate(zip(positions, types)):
            if i == target_idx or i in exclude_indices:
                continue
                
            pos = np.array(pos)
            r_vector = target_pos - pos
            r_magnitude = np.linalg.norm(r_vector)
            if r_magnitude < 1e-10:
                continue
                
            r_meter = r_magnitude * 1e-10
            symbol = self.atom_type_map[type_num]
            source_charge = abs(self.charges[symbol]) * self.e
            target_charge = abs(self.charges[target_symbol]) * self.e
            F_magnitude = (self.k * source_charge * target_charge) / (r_meter * r_meter)
            F_magnitude_angstrom = F_magnitude * 1e-10
            
            if symbol in ['Fe', 'H']:
                if target_symbol == 'O':
                    F_vector = -F_magnitude_angstrom * (r_vector / r_magnitude)
                else:
                    F_vector = F_magnitude_angstrom * (r_vector / r_magnitude)
            else:
                if target_symbol == 'O':
                    F_vector = F_magnitude_angstrom * (r_vector / r_magnitude)
                else:
                    F_vector = -F_magnitude_angstrom * (r_vector / r_magnitude)
                
            electric_force[i] = F_vector
                
        return electric_force

    def analyze_frame(self, frame):
        """Analyze a single frame of the trajectory."""
        positions = frame['positions']
        types = frame['types']
        
        try:
            fe_idx = types.index(3)  # Find the index of the Fe atom (type 3)
        except ValueError:
            return None
        
        frame_data = {'O_data': [], 'H_data': [], 'OH_data': []}
        
        for i, type_num in enumerate(types):
            if type_num == 1:  # Oxygen atom
                water_indices = self.get_water_indices(types, i)
                if not water_indices:
                    continue
                
                fe_dist = np.linalg.norm(np.array(positions[i]) - np.array(positions[fe_idx]))
                electric_field_o = self.calculate_electric_field(positions, types, i, water_indices)
                E_o = np.sum(list(electric_field_o.values()), axis=0)
                
                electric_force_o = self.calculate_electric_force(positions, types, i, water_indices)
                F_o = np.sum(list(electric_force_o.values()), axis=0)
                
                frame_data['O_data'].append({
                    'fe_dist': fe_dist,
                    'E_magnitude': np.linalg.norm(E_o),
                    'E_components': E_o,
                    'F_magnitude': np.linalg.norm(F_o),
                    'F_components': F_o
                })
                
                for h_idx in [i + 1, i + 2]:
                    electric_field_h = self.calculate_electric_field(positions, types, h_idx, water_indices)
                    E_h = np.sum(list(electric_field_h.values()), axis=0)
                    
                    electric_force_h = self.calculate_electric_force(positions, types, h_idx, water_indices)
                    F_h = np.sum(list(electric_force_h.values()), axis=0)
                    
                    frame_data['H_data'].append({
                        'fe_dist': np.linalg.norm(np.array(positions[h_idx]) - np.array(positions[fe_idx])),
                        'E_magnitude': np.linalg.norm(E_h),
                        'E_components': E_h,
                        'F_magnitude': np.linalg.norm(F_h),
                        'F_components': F_h
                    })
                    
                    oh_vector = np.array(positions[i]) - np.array(positions[h_idx])
                    oh_unit = oh_vector / np.linalg.norm(oh_vector)
                    
                    F_o_proj = np.dot(F_o, oh_unit)
                    F_h_proj = np.dot(F_h, oh_unit)
                    F_oh_net = F_o_proj + F_h_proj
                    
                    if F_o_proj > 0 and F_h_proj < 0:
                        oh_status = 'stretched'
                    elif F_o_proj > 0 and F_h_proj > 0:
                        if abs(F_o_proj) > abs(F_h_proj):
                            oh_status = 'stretched'
                        else:
                            oh_status = 'compressed'
                    elif F_o_proj < 0 and F_h_proj > 0:
                        oh_status = 'compressed'
                    elif F_o_proj < 0 and F_h_proj < 0:
                        if abs(F_h_proj) > abs(F_o_proj):
                            oh_status = 'stretched'
                        else:
                            oh_status = 'compressed'
                    else:
                        oh_status = 'undefined'
                    
                    frame_data['OH_data'].append({
                        'fe_dist': fe_dist,
                        'F_o_proj': F_o_proj,
                        'F_h_proj': F_h_proj,
                        'F_oh_net': F_oh_net,
                        'oh_status': oh_status
                    })
        
        return frame_data

    def plot_magnitude_distribution(self, all_data, data_key, value_key, xlabel, filename,show_legend=False):
        """Plot distributions of field or force magnitudes."""
        plt.figure(figsize=(10, 8))
        for shell_min, shell_max in self.shell_ranges:
            shell_data = [np.linalg.norm(d[value_key]) * 1e18 if 'F_' in value_key else np.linalg.norm(d[value_key]) 
                          for frame_data in all_data 
                          for d in frame_data[data_key]
                          if shell_min <= d['fe_dist'] < shell_max]
            if shell_data:
                weights = np.ones_like(shell_data) / len(shell_data)
                plt.hist(shell_data, bins=50, alpha=0.5, weights=weights,
                        label=f'Shell {shell_min}-{shell_max} Å')
        
        plt.xlabel(xlabel + (' (1e-18 eV/Å)' if 'F_' in value_key else ' (V/Å)'), fontsize=25, fontweight='bold')
        plt.ylabel('Probability', fontsize=25, fontweight='bold')
        plt.xticks(fontsize=25, fontweight='bold')
        plt.yticks(fontsize=25, fontweight='bold')
        plt.xlim(0, 10)     
        if show_legend:
            plt.legend()
        plt.savefig(os.path.join(self.output_dir, f'{filename}.pdf'),
                   dpi=400, bbox_inches='tight')
        plt.show()

    def plot_component_distribution(self, all_data, data_key, value_key, component_idx, title, filename):
        """Plot distributions of field or force components."""
        plt.figure(figsize=(10, 8))
        for shell_min, shell_max in self.shell_ranges:
            shell_data = [d[value_key][component_idx] * 1e18 if 'F_' in value_key else d[value_key][component_idx] 
                          for frame_data in all_data 
                          for d in frame_data[data_key]
                          if shell_min <= d['fe_dist'] < shell_max]
            if shell_data:
                weights = np.ones_like(shell_data) / len(shell_data)
                plt.hist(shell_data, bins=50, alpha=0.5, weights=weights,
                        label=f'Shell {shell_min}-{shell_max} Å')
        
        plt.xlabel(title + (' (1e-18 eV/Å)' if 'F_' in value_key else ' (V/Å)'), fontsize=25, fontweight='bold')
        plt.ylabel('Probability', fontsize=25, fontweight='bold')
        plt.xticks(fontsize=25, fontweight='bold')
        plt.yticks(fontsize=25, fontweight='bold')
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, f'{filename}.pdf'),
                   dpi=400, bbox_inches='tight')
        plt.show()

    def plot_oh_force_projections(self, all_data):
        """Plot distributions of force projections on O-H bond."""
        # Plot Oxygen Force Projection
        plt.figure(figsize=(10, 8))
        for shell_min, shell_max in self.shell_ranges:
            shell_data_o = [d['F_o_proj'] * 1e18 for frame_data in all_data 
                           for d in frame_data['OH_data']
                           if shell_min <= d['fe_dist'] < shell_max]
            if shell_data_o:
                weights = np.ones_like(shell_data_o) / len(shell_data_o)
                plt.hist(shell_data_o, bins=50, alpha=0.5, weights=weights,
                        label=f'Shell {shell_min}-{shell_max} Å')
        
        plt.xlabel('O Force on O-H Bond (1e-18 eV/Å)', fontsize=25, fontweight='bold')
        plt.ylabel('Probability', fontsize=25, fontweight='bold')
        plt.xticks(fontsize=25, fontweight='bold')
        plt.yticks(fontsize=25, fontweight='bold')

        #plt.legend()
        plt.savefig(os.path.join(self.output_dir, 'Oxygen_Force_Projection.pdf'),
                   dpi=400, bbox_inches='tight')
        plt.show()

        # Plot Hydrogen Force Projection
        plt.figure(figsize=(10, 8))
        for shell_min, shell_max in self.shell_ranges:
            shell_data_h = [d['F_h_proj'] * 1e18 for frame_data in all_data 
                           for d in frame_data['OH_data']
                           if shell_min <= d['fe_dist'] < shell_max]
            if shell_data_h:
                weights = np.ones_like(shell_data_h) / len(shell_data_h)
                plt.hist(shell_data_h, bins=50, alpha=0.5, weights=weights,
                        label=f'Shell {shell_min}-{shell_max} Å')
        
        plt.xlabel('H Force on O-H Bond (1e-18 eV/Å)', fontsize=25, fontweight='bold')
        plt.ylabel('Probability', fontsize=25, fontweight='bold')
        plt.xticks(fontsize=25, fontweight='bold')
        plt.yticks(fontsize=25, fontweight='bold')
        #plt.legend()
        plt.savefig(os.path.join(self.output_dir, 'Hydrogen_Force_Projection.pdf'),
                   dpi=400, bbox_inches='tight')
        plt.show()

        # Plot Net Force on O-H Bond
        plt.figure(figsize=(10, 8))
        for shell_min, shell_max in self.shell_ranges:
            shell_data_net = [d['F_oh_net'] * 1e18 for frame_data in all_data 
                             for d in frame_data['OH_data']
                             if shell_min <= d['fe_dist'] < shell_max]
            if shell_data_net:
                weights = np.ones_like(shell_data_net) / len(shell_data_net)
                plt.hist(shell_data_net, bins=50, alpha=0.5, weights=weights,
                        label=f'Shell {shell_min}-{shell_max} Å')
        
        plt.xlabel('Net Force on O-H Bond (1e-18 eV/Å)', fontsize=25, fontweight='bold')
        plt.ylabel('Probability', fontsize=25, fontweight='bold')
        #plt.legend()
        plt.savefig(os.path.join(self.output_dir, 'Net_Force_on_OH_Bond.pdf'),
                   dpi=400, bbox_inches='tight')
        plt.show()
    
def main():
    trajectory_file = 'dump.lammpstrj'  # Update with your trajectory file path
    analyzer = WaterShellAnalysis(trajectory_file)
    
    frames = analyzer.read_lammps_trajectory()
    all_data = []
    
    for frame in tqdm(frames):
        frame_data = analyzer.analyze_frame(frame)
        if frame_data:
            all_data.append(frame_data)
    
    # Plot distributions
    analyzer.plot_magnitude_distribution(all_data, 'O_data', 'E_components', 
                                      'Electric Field Magnitude at O', 'E_O_magnitude')
    analyzer.plot_magnitude_distribution(all_data, 'H_data', 'E_components', 
                                      'Electric Field Magnitude at H', 'E_H_magnitude')
    analyzer.plot_magnitude_distribution(all_data, 'O_data', 'F_components', 
                                      'Force Magnitude at O', 'F_O_magnitude')
    analyzer.plot_magnitude_distribution(all_data, 'H_data', 'F_components', 
                                      'Force Magnitude at H', 'F_H_magnitude')
    
    analyzer.plot_oh_force_projections(all_data)

if __name__ == "__main__":
    main()