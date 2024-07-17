import sys 
import pickle as pk
import sailfish
import numpy as np 
import matplotlib.pyplot as plt 
import multiprocessing as mp
from multiprocessing import Pool, cpu_count
from pathlib import Path
import os
import re
import fnmatch
import gc
import scipy

Plot = True

import time

def load_checkpoint(filename):
    with open(filename, "rb") as file:
        chkpt = pk.load(file)
        return chkpt


def CavityContour(chkpt):
	fields = {
	"sigma": lambda p: p[:, :, 0],
	"vx": lambda p: p[:, :, 1],
	"vy": lambda p: p[:, :, 2],
	"torque": None,
		}

	mesh   = chkpt["mesh"]
	prim   = chkpt["solution"]
	f      = fields["sigma"](prim).T
	extent = mesh.x0, mesh.x1, mesh.y0, mesh.y1	
	p = plt.contour(f,levels = [0.2],extent=extent).collections[0].get_paths()[0]
	v = p.vertices
	v_resampled = scipy.signal.resample(v, 500)
	return v_resampled


def MaxDist(points):

	Npoints = len(points)

	Dist_Vectors_i = np.zeros([Npoints,Npoints])
	Dist_Vectors_j = np.zeros([Npoints,Npoints])
	Distance       = np.zeros([Npoints,Npoints])
	
	for j in range(0,Npoints-2):
		for i in range(j+1,Npoints-1):
			xdist                = points[i,0] - points[j,0]
			ydist                = points[i,1] - points[j,1]
			Dist_Vectors_i[i][j] = xdist
			Dist_Vectors_j[i][j] = ydist
			Distance[i][j]       = np.sqrt(xdist**2+ydist**2)

	Semi_major_axis = np.max(Distance) / 2.
	k, l            = np.where(Distance == 2 * Semi_major_axis)
	Max_Slope       = np.arctan2(Dist_Vectors_j[k[0]][l[0]],Dist_Vectors_i[k[0]][l[0]])
	Minor_Distance  = np.zeros([Npoints,Npoints])
	for j in range(0,Npoints-2):
		for i in range(j+1,Npoints-1):
			proj_x = Dist_Vectors_i[i][j] * np.cos(Max_Slope+np.pi/2)
			proj_y = Dist_Vectors_j[i][j] * np.sin(Max_Slope+np.pi/2)
			Minor_Distance[i][j] = proj_x + proj_y

	Semi_minor_axis     = np.max(Minor_Distance) / 2.
	Cavity_Eccentricity = np.sqrt(1-(Semi_minor_axis/Semi_major_axis)**2)

	Properties = {
	"SemiMajorAxis": Semi_major_axis,
	"SemiMinorAxis": Semi_minor_axis,
	"Eccentricity":Cavity_Eccentricity,
	"Apsidal_X_Positions":[points[k,0],points[l,0]],
	"Apsidal_Y_Positions":[points[k,1],points[l,1]],
	"Cavity_Slope_Radians":Max_Slope,
	"Cavity_Slope_Degrees":Max_Slope*180/np.pi,
	}
	return Properties



def main_cbdiso_2d(chkpt,points,in_dir):
	fields = {
		"sigma": lambda p: p[:, :, 0],
		"vx": lambda p: p[:, :, 1],
		"vy": lambda p: p[:, :, 2],
		"torque": None,
		}

	fig, ax = plt.subplots(figsize=[12, 9])
	mesh = chkpt["mesh"]
	prim = chkpt["solution"]

	f = fields["sigma"](prim).T
	extent = mesh.x0, mesh.x1, mesh.y0, mesh.y1
	cm = ax.imshow(
		np.log10(f),
		origin="lower",
		vmin=-5,
		vmax=1,
		cmap="magma",
		extent=extent,
		)

	CavityProperties = MaxDist(points)
	a_cavity = CavityProperties["SemiMajorAxis"]
	e_cavity = CavityProperties["Eccentricity"]
	apses_x  = CavityProperties["Apsidal_X_Positions"]
	apses_y  = CavityProperties["Apsidal_Y_Positions"]
	arg_aps  = CavityProperties["Cavity_Slope_Radians"]

	def Parameterisation(a,e,f,w):
		r = a * (1-e**2) / (1+e*np.cos(f-w))
		return apses_x[0] - a*(1-e)*np.cos(w) + r * (np.cos(f)), apses_y[0] -a*(1-e)*np.sin(w) + r * (np.sin(f))
		
	Position_List = np.array([Parameterisation(a_cavity,e_cavity,i,arg_aps) for i in np.linspace(0,2*np.pi,100)]).reshape(100,2)
	plt.plot(Position_List[:,0],Position_List[:,1],c = 'black', linestyle = 'dashed', linewidth = 2)
	
	fig.suptitle('Orbit %g'%(chkpt["time"] / 2 / np.pi))
	FigDirectory =  in_dir
	ax.set_xlim(-4, 4)
	ax.set_ylim(-4, 4)
	plt.savefig(FigDirectory + '/CavityFit_%g.png'%(chkpt["time"] / 2 / np.pi), dpi = 400)


def MP_Cavity_Properties(arg,in_dir):
	chkpt             = load_checkpoint(arg)
	contour_lines     = CavityContour(chkpt)


	cavity_properties = MaxDist(contour_lines)
	Binary_SMA        = np.array([s[ 1] for s in chkpt['timeseries']])[-1]
	cavity_properties["Binary_SemiMajorAxis"] = Binary_SMA
	cavity_properties["CurrentTime"]          = chkpt["time"] / 2 / np.pi
	cavity_properties["viscosity"]            = chkpt["model_parameters"]["nu"]
	cavity_properties["Retrograde"]           = chkpt["model_parameters"]["retrograde"]
	
	if Plot:
		main_cbdiso_2d(chkpt,contour_lines,in_dir)
	return cavity_properties


def FitCavityCheck(in_dir):
	
	directory    = Path(in_dir)
	Fit_Cavities = True
	nu           = False

	for i in os.listdir(directory):
		if fnmatch.fnmatch(in_dir + '/' + i, '*.pk'):
			if fnmatch.fnmatch(in_dir + '/' + i, 'CavityProperties_nu.*.pk'):
				print('We have found pk file with pattern')
				Fit_Cavities = False
				break
			else:
				pass

			while nu == False:
				nu = load_checkpoint(in_dir + '/' + i)['model_parameters']['nu']
			
		else:
			pass
	print('Are we gonna run a cavity fit for',in_dir,'?',Fit_Cavities)
	print('Looking in',os.listdir(directory))

	return Fit_Cavities, nu


def CavityEvolution(in_dir):

	Fit_Cavities, nu = FitCavityCheck(in_dir)
	CavityFileName   = in_dir + '/CavityProperties_nu.%g.pk'%(nu)

	if Fit_Cavities:
		Time_Snapshots  = []
		Semi_Major_Axis = []
		Eccentricity    = []
		Argument_Apses  = []
		BinarySMA       = []

		for file in os.listdir(in_dir):

			if fnmatch.fnmatch(in_dir + '/' + file, '*chkpt*.pk'):
				
				cav_props = MP_Cavity_Properties(in_dir + '/' + file, in_dir)
				
				Time_Snapshots.append(cav_props["CurrentTime"])
				Semi_Major_Axis.append(cav_props["SemiMajorAxis"])
				Eccentricity.append(cav_props["Eccentricity"])
				Argument_Apses.append(cav_props["Cavity_Slope_Radians"])
				BinarySMA.append(cav_props["Binary_SemiMajorAxis"])

			else:
				pass


		lists         = list(zip(Time_Snapshots, Semi_Major_Axis, Eccentricity, Argument_Apses,BinarySMA))
		sorted_lists  = sorted(lists, key=lambda x: x[0])

		sorted_times, sorted_SMA, sorted_ecc, sorted_Apses, sorted_Binary_SMA = zip(*sorted_lists)

		Cavity = {
		'Timeseries':sorted_times,
		'SemiMajor_Axis':sorted_SMA,
		'Eccentricity':sorted_ecc,
		'Inclination':sorted_Apses,
		'Binary_SMA':sorted_Binary_SMA,
		'nu':nu,
		'Retrograde':cav_props['Retrograde'],
		}

		with open(CavityFileName, "wb") as cvt:
			pk.dump(Cavity, cvt)

	else:
		pass

	return CavityFileName

def Load_Cavity_Files(parent_dir):
	subdirs = [parent_dir + '/' + name for name in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, name))]
	
	for in_dir in subdirs:
		
		CavityFileName = CavityEvolution(in_dir)
		Cavity         = load_checkpoint(parent_dir + in_dir + CavityFileName)
		



		# extract nu from child direct
		# load pk file saved with nu
		# save pk and nu to dict

	# return all dictionaries


Load_Cavity_Files(sys.argv[1])



"""

fig, ax = plt.subplots(figsize=[12, 9])
plt.title('Cavity Properties')
plt.plot(sorted_times,sorted_SMA, c = 'brown', linewidth = 2, label = r'Semi Major Axis $[a_0]$')
plt.plot(sorted_times,sorted_ecc, c = 'blue', linestyle = 'dashed',linewidth = 2, label = 'Eccentricity')
plt.plot(sorted_times,sorted_Apses, c = 'silver', linestyle = 'dotted',linewidth = 2, label = 'Apsidal Inclination (Radians)')
plt.scatter(sorted_times,sorted_SMA, c = 'brown', s = 50)
plt.scatter(sorted_times,sorted_ecc, c = 'blue', s = 50)
plt.scatter(sorted_times,sorted_Apses, c = 'silver', s = 50)
plt.xlabel(r'Time $2\pi\Omega_0^{-1}$')
plt.legend()
FigDirectory =  sys.argv[1]
pngname = FigDirectory + f"{'/CavityProperties'}.{int(sorted_times[-1]):04d}.png"
fig.savefig(pngname, dpi=400)


fig, ax = plt.subplots(figsize=[12, 9])
plt.title('Cavity Semi Major Axis',fontsize=25)
plt.xlabel(r'$\log_{10} a_\mathrm{bin}~[a_0]$',fontsize=20)
plt.ylabel(r'$\log_{10} a_\mathrm{cav}~[a_0]$',fontsize=20)
plt.plot(np.array(sorted_Binary_SMA),np.array(sorted_SMA),c='black')
plt.scatter(np.array(sorted_Binary_SMA),np.array(sorted_SMA),c='black',marker='*')
plt.gca().invert_xaxis()
ax.set_yscale('log')
ax.set_xscale('log')
plt.yticks([10,1,0.1,0.001])
plt.xticks([1,0.1,0.01])
plt.savefig(FigDirectory + '/Decoupling.png', dpi=400)
"""

exit()
