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

Plot = True

import time

def load_checkpoint(filename):
    start = time.time()
    with open(filename, "rb") as file:
        chkpt = pk.load(file)
        print('Importing file took',time.time()-start)
        return chkpt


def CavityContour(chkpt):
	start = time.time()
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
	print('CavityContour took',time.time()-start)
	return v


def MaxDist(points):
	start = time.time()

	Npoints = len(points)
	print('Number of points is',Npoints)

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
	print('Maxdist took', time.time()-start)
	return Properties



def main_cbdiso_2d(chkpt,points):
	start = time.time()
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
	FigDirectory =  sys.argv[1]
	ax.set_xlim(-4, 4)
	ax.set_ylim(-4, 4)
	plt.savefig(FigDirectory + '/CavityFit_%g.png'%(chkpt["time"] / 2 / np.pi), dpi = 400)
	print('CBDISO took', time.time()-start)


def MP_Cavity_Properties(arg):
	start = time.time()
	chkpt             = load_checkpoint(arg)

	contour_lines     = CavityContour(chkpt)


	cavity_properties = MaxDist(contour_lines)
	Binary_SMA        = np.array([s[ 1] for s in chkpt['timeseries']])[-1]
	cavity_properties["Binary_SemiMajorAxis"] = Binary_SMA
	cavity_properties["CurrentTime"]          = chkpt["time"] / 2 / np.pi
	cavity_properties["viscosity"]            = chkpt["model_parameters"]["nu"]
	if Plot:
		main_cbdiso_2d(chkpt,contour_lines)
	print('MP_Cavity_Properties done', time.time()-start)
	return cavity_properties


if __name__ == "__main__":
	
	directory = Path(sys.argv[1])

	for i in os.listdir(directory):
		if fnmatch.fnmatch(sys.argv[1] + '/' + i, '*.pk'):
			nu        = load_checkpoint(sys.argv[1] + '/' + i)['model_parameters']['nu']
			break
		else:
			pass
	

	Time_Snapshots  = []
	Semi_Major_Axis = []
	Eccentricity    = []
	Argument_Apses  = []
	BinarySMA       = []

	for file in os.listdir(directory):

		if fnmatch.fnmatch(sys.argv[1] + '/' + file, '*chkpt*.pk'):
			
			cav_props = MP_Cavity_Properties(sys.argv[1] + '/' + file)
			
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
	'Binary_SMA':sorted_Binary_SMA
	}


	CavityFileName = sys.argv[1] + '/CavityProperties_nu.%g.pk'%(nu)
	with open(CavityFileName, "wb") as cvt:
		pk.dump(Cavity, cvt)

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

	

	exit()
