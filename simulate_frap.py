#!/usr/bin/python3
import numpy as np
import scipy.ndimage as ndimage
import scipy.integrate

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.colors

import time
import math
import csv
import os

fe_diffusion_kernel = np.array(
         [[[0, 0, 0],
           [0, 1, 0],
           [0, 0, 0]],
            
          [[0,  1, 0],
           [1, -6, 1],
           [0,  1, 0]],
            
          [[0, 0, 0],
           [0, 1, 0],
           [0, 0, 0]]]);

def convolve(domain, kernel, dr, bcs=None):
    if bcs is None:
        return ndimage.convolve(domain, kernel / (dr ** 2), mode='constant', cval=1)
    elif bcs == 'noflux':
        return ndimage.convolve(domain, kernel / (dr ** 2), mode='nearest')
    else:
        raise RuntimeError('Invalid BC!')
def convolve_1d(domain, kernel, shape, dr, bcs=None):
    return convolve(domain.reshape(shape),kernel,dr,bcs).reshape((-1,))

def diffusion_two_phase(domain, phase_mask, partition_coeff, d_interior, d_exterior, dr,bcs=None):
    # To simulate a partition coefficent, divide inner phase by partition_coeff
    # before doing convolution
    modified_domain = domain.copy();
    modified_domain[phase_mask] /= partition_coeff;
    
    result = convolve(modified_domain, fe_diffusion_kernel, dr, bcs)
    # Scale back changes in interior by partition_coeff
    result[phase_mask] *= partition_coeff * d_interior;
    result[~phase_mask] *= d_exterior;
    return result;

def diffusion_1d(domain, phase_mask, partition_coeff, d_interior, d_exterior, dr, shape, bcs=None):
    return diffusion_two_phase(domain.reshape(shape), phase_mask, partition_coeff,
                               d_interior, d_exterior, dr,bcs).reshape((-1,))

def generate_packed_sphere_mask(domain_shape, num_spheres, r_sphere, r_centers):
    # Generates a hexagonal close packing of spheres, separated by r_centers
    num_rows = math.ceil(math.sqrt(num_spheres))
    entries_per_row = math.ceil(num_spheres / num_rows)
    horz_sep = r_centers;
    horz_offset = int(horz_sep / 2);
    vert_offset = int(math.sqrt(3) / 2 * r_centers);
    
    x_offsets = [horz_sep * i for i in range(entries_per_row)];
    avg_offset = sum(x_offsets) / entries_per_row;
    x_offsets.sort(key=lambda x: abs(x - avg_offset))
    
    x_start = domain_shape[0] / 2 - (horz_sep * (-1/2 + entries_per_row / 2));
    y_start = domain_shape[1] / 2 - (vert_offset * (-1/2 + num_rows / 2));
    
    total_mask = np.zeros(domain_shape, dtype=bool)
    n_placed = 0;
    for i in range(num_rows):
        for x_offset in x_offsets:
            if n_placed < num_spheres:
                total_mask |= generate_sphere_mask(domain_shape,
                           int(x_offset + x_start + (horz_offset * (i % 2))),
                           int(y_start + vert_offset * i),
                           int(domain_shape[2] / 2),
                           r_sphere);
            n_placed += 1
    return total_mask

def generate_sphere_mask(domain_shape,x,y,z,r):
    """
    Given an (x,y,z) sphere point (in index units) and r (in index units),
    returns a boolean mask over the domain where true values lie within the sphere.
    """
    x_vals = np.arange(domain_shape[0]).reshape(-1,1,1);
    y_vals = np.arange(domain_shape[1]).reshape(1,-1,1);
    z_vals = np.arange(domain_shape[2]).reshape(1,1,-1);
    return ((x_vals - x)**2 + (y_vals - y)**2 + (z_vals - z)**2) < r**2

def generate_cylinder_mask(domain_shape, x, y, r):
    """
    Given an (x,y) center of a circle, generates a cylindrical mask
    that covers the entire z dimension
    """
    x_vals = np.arange(domain_shape[0]).reshape(-1,1,1);
    y_vals = np.arange(domain_shape[1]).reshape(1,-1,1);
    z_vals = np.zeros((1,1,domain_shape[2]));
    return ((x_vals - x)**2 + (y_vals - y)**2 + z_vals) < r**2

def run_simulation(diffusion_interior, diffusion_exterior, partition_coeff,
                 phase_mask, bleach_mask, t_max, domain_height, domain_shape,
                 bcs=None):
    """
    Uses a 
    Inputs:
        
        domain_height and diffusion_interior/diffusion_exterior must be in the same units!
        
    """
    #width_spheres = int(domain_resolution * .7)
    ics = np.ones(domain_shape);
    ics[phase_mask] *= partition_coeff;
    ics[bleach_mask] = 0;
   #sphere_mask = generate_packed_sphere_mask(ics.shape, num_spheres, width_spheres / 2, int(1.3 *width_spheres))
    
    dr = domain_height / domain_shape[2];
    def ivp_func(t,x,mask=phase_mask,shape=domain_shape,
                 di=diffusion_interior, de=diffusion_exterior, dr=dr, K=partition_coeff):
        return diffusion_1d(x, mask, K, di, de, dr, shape, bcs)
    
    results = scipy.integrate.solve_ivp(ivp_func, [0, t_max], ics.reshape((-1,)), method='RK23',
                                        t_eval=np.linspace(0, t_max, 500));
    assert(results.success);
    return (results.t,results.y.reshape(domain_shape + (results.y.shape[1],)))

def post_process(times, domain_solution, bleach_mask, options):
    """
    Options is expected as a dictionary of key/value pairs
    """
    field_names = sorted(options.keys())
    
    success = False
    runs_postfix = ''
    while not success:
        if not os.path.isfile('runs' + runs_postfix + '.csv'):
            success = True
            with open('runs' + runs_postfix + '.csv', 'w') as runs:
                writer = csv.DictWriter(runs, field_names)
                writer.writeheader()
            continue
        else:
            with open('runs' + runs_postfix + '.csv', 'r') as runs:
                reader = csv.DictReader(runs)
                if field_names != reader.fieldnames:
                    print('Warning: field mismatch in preexisting runs file! Adding a postfix')
                    if runs_postfix == '':
                        runs_postfix = '_1'
                    else:
                        runs_postfix = '_' + str(int(runs_postfix[1:]) + 1)
                else:
                    success = True
    mean_recovery = [np.mean((domain_solution[:,:,:,i])[bleach_mask]) for i in range(domain_solution.shape[3])]
    
    if not os.path.exists('run_data' + runs_postfix):
        os.mkdir('run_data' + runs_postfix)
    postfix = ''
    while os.path.isfile(os.path.join('run_data' + runs_postfix, options['filename'] + postfix + '.csv')):
        print('Warning: filename {} already existed! Iterating the filename'.format(
            options['filename'] + postfix))
        if postfix == '':
            postfix = '_1'
        else:
            postfix = '_' + str(int(postfix[1:]) + 1)
    options['filename'] += postfix
    with open('runs' + runs_postfix + '.csv', 'a') as runs, open(
        os.path.join('run_data' + runs_postfix, options['filename'] + '.csv'), 'w') as data:
        runs_writer = csv.DictWriter(runs, field_names)
        data_writer = csv.writer(data)
        runs_writer.writerow(options);
        data_writer.writerow(['t', 'Mean concentration'])
        for time, mean in zip(times, mean_recovery):
            data_writer.writerow([time, mean])
        
        # Now save a picture of the domain
        cmap = plt.cm.gray
        norm = plt.Normalize(vmin=0, vmax=np.max(domain_solution))
        bleach_norm = plt.Normalize()
        image = cmap(norm(domain_solution[:,:,int(domain_solution.shape[2] / 2),0]))
        final_image = cmap(norm(domain_solution[:,:,int(domain_solution.shape[2] / 2),-1]))
        bleach_image = cmap(bleach_norm(bleach_mask[:,:,int(bleach_mask.shape[2] / 2)]))
        plt.imsave(os.path.join('run_data' + runs_postfix, options['filename'] + '_start.png'),image)
        plt.imsave(os.path.join('run_data' + runs_postfix, options['filename'] + '_end.png'),final_image)
        plt.imsave(os.path.join('run_data' + runs_postfix, options['filename'] + '_bleach.png'),bleach_image)
        
def run_1d_sphere_experiment(
    d_interior, d_exterior, partition_coefficent,
    sphere_diameter, bleach_diameter, t_max, resolution, filename):
    """Given diffusion coefficents measured in square microns and diameters in microns"""
    domain_shape = (resolution * 5, resolution * 5, resolution)
    # make the domain height 1 micron
    domain_height = 1;
    sphere_index_radius = int(resolution * sphere_diameter / 2)
    bleach_index_radius = int(resolution * bleach_diameter / 2)
    
    phase_mask = generate_sphere_mask(domain_shape,
         int(domain_shape[0] / 2), int(domain_shape[1] / 2), int(domain_shape[2] / 2), sphere_index_radius)
    bleach_mask = generate_cylinder_mask(domain_shape,
         int(domain_shape[0] / 2), int(domain_shape[1] / 2), bleach_index_radius)
    
    ics = np.ones(domain_shape);
    ics[phase_mask] *= partition_coefficent;
    normalization = np.mean(ics[bleach_mask])
    
    times, domain_solution = run_simulation(d_interior, d_exterior, partition_coefficent,
           phase_mask, bleach_mask, t_max, domain_height, domain_shape)
    post_process(times, domain_solution, bleach_mask,
        {'filename': filename, 'd_interior (um^2 / s)': d_interior, 'd_exterior (um^2 / s)': d_exterior
        ,'partition': partition_coefficent, 'sphere_diameter (um)': sphere_diameter
        ,'bleach_diameter (um)': bleach_diameter, 't_max (s)': t_max, 'mesh_resolution': resolution
        ,'normalization':normalization})

if __name__ == '__main__':
    pre_wall, pre_cpu = (time.perf_counter(), time.process_time())
    for idx1, diameter in enumerate(np.arange(.7,.1,-.2)):
        for idx2, d_exterior in enumerate([1,10]):
            for idx3, d_interior in enumerate([.1,.01]):
                    for idx4, k in enumerate([2, 10, 50, 100]):
                        run_1d_sphere_experiment(d_interior, d_exterior, k, diameter, diameter, .3 / d_interior, 45,
                                 'equal_bleach_sphere_diameter_{}_{}_{}_{}'.format(idx1,idx2,idx3,idx4))
                        print('.',flush=True,end="")
    print('\n')
    post_wall, post_cpu = (time.perf_counter(), time.process_time())
    print('Wall time: {:.2f} sec, CPU time: {:.2f}'.format(post_wall - pre_wall, post_cpu - pre_cpu))