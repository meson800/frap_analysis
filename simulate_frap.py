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
import random
import itertools

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
    x_vals = []
    y_vals = []
    cylinder_masks = []
    sphere_masks = []
    n_placed = 0;
    for i in range(num_rows):
        for x_offset in x_offsets:
            if n_placed < num_spheres:
                x_val = int(x_offset + x_start + (horz_offset * (i % 2)))
                y_val = int(y_start + vert_offset * i)
                z_val = int(domain_shape[2] / 2)
                x_vals.append(x_val)
                y_vals.append(y_val)
                total_mask |= generate_sphere_mask(domain_shape,
                           x_val, y_val, z_val, r_sphere)
                cylinder_masks.append(generate_cylinder_mask(domain_shape,
                            x_val, y_val, r_sphere))
                sphere_masks.append(generate_sphere_mask(domain_shape,
                            x_val, y_val, z_val, r_sphere))
            n_placed += 1
    
    return (total_mask, np.sqrt(np.power(np.array(x_vals) ,2) +
                                np.power(np.array(y_vals), 2)),
            np.tan(np.array(x_vals) / np.array(y_vals)), cylinder_masks, sphere_masks)

def generate_bootstrap_sphere_mask(domain_shape, domain_height, r_bleach, num_spheres):
    # First import condensate size
    with open('condensate_size.csv') as f:
        condensate_diameter = [float(l) / 1000 for l in f]
    
    # Monte carlo sample until we find a packing that works
    converged = False
    while ~converged:
        sizes = np.array(random.choices(condensate_diameter, k=num_spheres))
        radius_samples = np.sqrt(np.random.uniform(0, r_bleach**2, num_spheres));
        angle_samples = np.random.uniform(0, 2 * np.pi, num_spheres);
        x_samples = radius_samples * np.cos(angle_samples);
        y_samples = radius_samples * np.sin(angle_samples);
        
        # check for rejection. Reject if sphere falls outside bleach radius
        converged = np.all(radius_samples + sizes / 2 <= r_bleach);
        
        # Check to make sure spheres do not overlap
        for (i,j) in itertools.product(range(num_spheres), repeat=2):
            if i == j:
                continue
            intra_distance = np.sqrt((x_samples[i] - x_samples[j])**2 + (y_samples[i] - y_samples[j])**2)
            if  intra_distance < (sizes[i] + sizes[j]) / 2:
                    converged = False
    # We have converged!
    total_mask = np.zeros(domain_shape, dtype=bool)
    real_to_idx = domain_shape[2] / domain_height;
    cylinder_masks = []
    sphere_masks = []
    for diameter, x, y in zip(sizes, x_samples, y_samples):
        total_mask |= generate_sphere_mask(domain_shape,
                   int(domain_shape[0] / 2 + (x * real_to_idx)),
                   int(domain_shape[1] / 2 + (y * real_to_idx)),
                   int(domain_shape[2] / 2),
                   int(diameter * real_to_idx / 2))
        cylinder_masks.append(generate_cylinder_mask(domain_shape,
                    int(domain_shape[0] / 2 + (x * real_to_idx)),
                    int(domain_shape[1] / 2 + (y * real_to_idx)),
                    int(diameter * real_to_idx / 2)))
        sphere_masks.append(generate_sphere_mask(domain_shape,
                   int(domain_shape[0] / 2 + (x * real_to_idx)),
                   int(domain_shape[1] / 2 + (y * real_to_idx)),
                   int(domain_shape[2] / 2),
                   int(diameter * real_to_idx / 2)))
    return (total_mask, sizes, radius_samples, angle_samples, cylinder_masks, sphere_masks)

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
    normalization_const = np.mean(ics[bleach_mask])
    ics[bleach_mask] = 0;
   #sphere_mask = generate_packed_sphere_mask(ics.shape, num_spheres, width_spheres / 2, int(1.3 *width_spheres))
    
    dr = domain_height / domain_shape[2];
    def ivp_func(t,x,mask=phase_mask,shape=domain_shape,
                 di=diffusion_interior, de=diffusion_exterior, dr=dr, K=partition_coeff):
        return diffusion_1d(x, mask, K, di, de, dr, shape, bcs)
    
    def terminate_cond(t, y):
        print('{:.3f}'.format(np.mean(y.reshape(domain_shape)[bleach_mask]) / normalization_const))
        return (np.mean(y.reshape(domain_shape)[bleach_mask]) / normalization_const) - .9
    terminate_cond.terminal = True
    
    results = scipy.integrate.solve_ivp(ivp_func, [0, t_max], ics.reshape((-1,)), method='RK23',
                                        t_eval=np.linspace(0, t_max, 600), events=terminate_cond);
    assert(results.success);
    return (results.t,results.y.reshape(domain_shape + (results.y.shape[1],)))

def post_process(times, domain_solution, phase_mask, bleach_mask,
                 average_masks, average_names, norms, options, video=False):
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
    
    recoveries = np.array([[np.mean((domain_solution[:,:,:,i])[mask]) / norms[j]
                            for i in range(domain_solution.shape[3])]
                     for j, mask in enumerate(average_masks)])
    
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
        data_writer.writerow(['t'] + average_names)
        for i, time in enumerate(times):
            data_writer.writerow([time] + list(recoveries[:,i]))
        
        # Now save a picture of the domain
        cmap = plt.cm.gray
        norm = plt.Normalize(vmin=0, vmax=np.max(domain_solution))
        bleach_norm = plt.Normalize()
        phase_norm = plt.Normalize()
        image = cmap(norm(domain_solution[:,:,int(domain_solution.shape[2] / 2),0]))
        final_image = cmap(norm(domain_solution[:,:,int(domain_solution.shape[2] / 2),-1]))
        bleach_image = cmap(bleach_norm(bleach_mask[:,:,int(bleach_mask.shape[2] / 2)]))
        phase_image = cmap(phase_norm(phase_mask[:,:,int(phase_mask.shape[2] / 2)]))
        plt.imsave(os.path.join('run_data' + runs_postfix, options['filename'] + '_start.png'),image)
        plt.imsave(os.path.join('run_data' + runs_postfix, options['filename'] + '_end.png'),final_image)
        plt.imsave(os.path.join('run_data' + runs_postfix, options['filename'] + '_bleach.png'),bleach_image)
        plt.imsave(os.path.join('run_data' + runs_postfix, options['filename'] + '_phase.png'),phase_image)

        if video:
            fig, axes = plt.subplots(1,2, figsize=(15,5))
            midpoint = int(domain_solution.shape[2] / 2)
            im = axes[0].imshow(domain_solution[:,:,midpoint,0], 'gray',matplotlib.colors.Normalize(0,2), aspect='equal')
            recovery_curves = axes[1].plot(times, recoveries.T)
            axes[1].set_title('Recovery')
            point, = axes[1].plot(0,recoveries[0,0], 'ko')
            axes[1].plot(times, recoveries.T, 'k', linewidth=.5, alpha=.4)
            axes[1].legend(average_names)
            def animate_func(i):
                im.set_array(domain_solution[:,:,midpoint,i])
                for j in range(len(recovery_curves)):
                    recovery_curves[j].set_xdata(times[:i])
                    recovery_curves[j].set_ydata(recoveries[j,:i])
                point.set_data(times[i], recoveries[0,i])
            anim = FuncAnimation(fig,animate_func,frames=domain_solution.shape[3])
            anim.save(os.path.join('run_data' + runs_postfix, options['filename'] + '_video.mp4'))
            plt.close(fig)
        
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
    post_process(times, domain_solution, phase_mask, bleach_mask,
        {'filename': filename, 'd_interior (um^2 / s)': d_interior, 'd_exterior (um^2 / s)': d_exterior
        ,'partition': partition_coefficent, 'sphere_diameter (um)': sphere_diameter
        ,'bleach_diameter (um)': bleach_diameter, 't_max (s)': t_max, 'mesh_resolution': resolution
        ,'normalization':normalization})
    
def run_multi_sphere_experiment(
    d_interior, d_exterior, partition_coefficent,
    num_spheres, bleach_radius, t_max, resolution, filename, video=False):
    """Given diffusion coefficents measured in square microns and diameters in microns"""
    domain_shape = (resolution * 3, resolution * 3, resolution)
    # make the domain height 1 micron
    domain_height = 1;
    
    bleach_index_radius = int(resolution * bleach_radius)
    
    phase_mask, sizes, radii, angles, cylinders, spheres = generate_bootstrap_sphere_mask(domain_shape,
                              domain_height, bleach_radius, num_spheres)
    
    bleach_mask = generate_cylinder_mask(domain_shape,
         int(domain_shape[0] / 2), int(domain_shape[1] / 2), bleach_index_radius)
    
    ics = np.ones(domain_shape);
    ics[phase_mask] *= partition_coefficent;
    normalizations = [np.mean(ics[mask]) for mask in [bleach_mask] + cylinders + spheres]
    
    pre_wall, pre_cpu = (time.perf_counter(), time.process_time())
    times, domain_solution = run_simulation(d_interior, d_exterior, partition_coefficent,
           phase_mask, bleach_mask, t_max, domain_height, domain_shape)
    post_wall, post_cpu = (time.perf_counter(), time.process_time())
    post_process(times, domain_solution, phase_mask, bleach_mask,
                 [bleach_mask] + cylinders + spheres, 
                 ['Bleach spot'] + ['C{}'.format(i) for i in range(len(cylinders))] +
                 ['S{}'.format(i) for i in range(len(cylinders))],
                 normalizations,
        {'filename': filename, 'd_interior (um^2 / s)': d_interior, 'd_exterior (um^2 / s)': d_exterior
        ,'partition': partition_coefficent, 'number_spheres': num_spheres,
         'sphere_sizes (um)': str(sizes), 'sphere_center_radius (um)': str(radii),
         'sphere_center_angle (radians)': str(angles)
        ,'bleach_radius (um)': bleach_radius, 't_max (s)': t_max, 'mesh_resolution': resolution
        ,'normalizations':normalizations,
         'wall_time (s)': post_wall - pre_wall, 'cpu_time (s)': post_cpu - pre_cpu}, video)
    
def run_equal_multi_sphere_experiment(
    d_interior, d_exterior, partition_coefficent,
    bleach_radius, num_spheres, sphere_radius, sphere_spacing, 
    t_max, resolution, filename, video=False):
    """Given diffusion coefficents measured in square microns and diameters in microns"""
    domain_shape = (resolution * 3, resolution * 3, resolution)
    # make the domain height 1 micron
    domain_height = 1;
    
    bleach_index_radius = int(resolution * bleach_radius)
    sphere_index_radius = int(resolution * sphere_radius)
    sphere_index_spacing = int(resolution * sphere_spacing)
    
    phase_mask, radii, angles, cylinders, spheres = generate_packed_sphere_mask(domain_shape,
                              num_spheres, sphere_index_radius, sphere_index_spacing)
    radii /= resolution;
    sizes = np.ones((num_spheres,)) * sphere_radius * 2;
    
    bleach_mask = generate_cylinder_mask(domain_shape,
         int(domain_shape[0] / 2), int(domain_shape[1] / 2), bleach_index_radius)
    
    ics = np.ones(domain_shape);
    ics[phase_mask] *= partition_coefficent;
    normalizations = [np.mean(ics[mask]) for mask in [bleach_mask] + cylinders + spheres]
    
    pre_wall, pre_cpu = (time.perf_counter(), time.process_time())
    times, domain_solution = run_simulation(d_interior, d_exterior, partition_coefficent,
           phase_mask, bleach_mask, t_max, domain_height, domain_shape)
    post_wall, post_cpu = (time.perf_counter(), time.process_time())
    post_process(times, domain_solution, phase_mask, bleach_mask,
                 [bleach_mask] + cylinders + spheres, 
                 ['Bleach spot'] + ['C{}'.format(i) for i in range(len(cylinders))] +
                 ['S{}'.format(i) for i in range(len(cylinders))],
                 normalizations,
        {'filename': filename, 'd_interior (um^2 / s)': d_interior, 'd_exterior (um^2 / s)': d_exterior
        ,'partition': partition_coefficent, 'number_spheres': num_spheres,
         'sphere_sizes (um)': str(sizes), 'sphere_center_radius (um)': str(radii),
         'sphere_center_angle (radians)': str(angles)
        ,'bleach_radius (um)': bleach_radius, 't_max (s)': t_max, 'mesh_resolution': resolution
        ,'normalizations':normalizations,
         'wall_time (s)': post_wall - pre_wall, 'cpu_time (s)': post_cpu - pre_cpu}, video)
   
def single_expt():
    pre_wall, pre_cpu = (time.perf_counter(), time.process_time())
    for idx1, diameter in enumerate(np.arange(.7,.1,-.2)):
        for idx2, d_exterior in enumerate([10]):
            for idx3, d_interior in enumerate([.1,.01]):
                    for idx4, k in enumerate([2, 10, 50, 100]):
                        run_1d_sphere_experiment(d_interior, d_exterior, k, diameter, diameter, .15 / d_interior, 45,
                                 'equal_bleach_sphere_diameter_{}_{}_{}_{}'.format(idx1,idx2,idx3,idx4))
                        print('.',flush=True,end="")
    print('\n')
    post_wall, post_cpu = (time.perf_counter(), time.process_time())
    print('Wall time: {:.2f} sec, CPU time: {:.2f}'.format(post_wall - pre_wall, post_cpu - pre_cpu))
    
def run_random_expt():
    for repeat in itertools.count(start=0, step=1):
        for num_spheres in [9, 6, 3]:
            run_multi_sphere_experiment(.1, 1, 50, num_spheres, 1, .15, 55, '{}_spheres_repeat_{}'.format(num_spheres, repeat), True)
    
if __name__ == '__main__':
    for num_spheres in [9, 6, 3]:
        for sphere_diameter in [.1, .2, .3, .4]:
            for sphere_spacing in np.arange(sphere_diameter + .05, .45, .1):
                run_equal_multi_sphere_experiment(.1, 1, 50, 1, num_spheres,
                                 sphere_diameter / 2, sphere_spacing, .15, 55,
                                '{}_spheres_diameter_{}_{}'.format(num_spheres, sphere_diameter, sphere_spacing), True)