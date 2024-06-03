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

def load_checkpoint(filename):
    print("About to import",filename)
    with open(filename, "rb") as file:
        chkpt = pk.load(file)
        print("This file has been imported",filename)
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
	return v


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



def main_cbdiso_2d(chkpt,points):
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
	FigDirectory =  sys.argv[2]
	print('Saving Plot')
	plt.savefig(FigDirectory + '/CavityFit_%g.png'%(chkpt["time"] / 2 / np.pi), dpi = 400)



def MP_Cavity_Properties(arg):
	chkpt             = load_checkpoint(arg)
	contour_lines     = CavityContour(chkpt)
	cavity_properties = MaxDist(contour_lines)
	Binary_SMA        = np.array([s[ 1] for s in chkpt['timeseries']])[-1]
	cavity_properties["Binary_SemiMajorAxis"] = Binary_SMA
	cavity_properties["CurrentTime"]          = chkpt["time"] / 2 / np.pi

	if Plot:
		main_cbdiso_2d(chkpt,contour_lines)
	
	FileName = arg.replace('chkpt','CavityProperties')
	with open(FileName, "wb") as cvt:
		pk.dump(cavity_properties, cvt)

	del chkpt
	del contour_lines
	del cavity_properties
	del Binary_SMA

	gc.collect()
	print('Completed Task')
	

def CheckForCavityFileExistence():
	directory      = Path(sys.argv[1])
	chkpt_pattern  = re.compile(r'chkpt\.(\d{4})\.pk')
	cavity_pattern = 'CavityProperties.{:04d}.pk'
	Missing_cavity = []

	for filename in os.listdir(directory):
		match = chkpt_pattern.match(filename)
		if match:
			number = int(match.group(1))
			corresponding_file = cavity_pattern.format(number)
			full_path = os.path.join(directory, filename)
			if os.path.exists(os.path.join(directory, corresponding_file)):
				pass
			else:
				Missing_cavity.append(full_path)

	if Missing_cavity == []:
		return True
	else:
		return Missing_cavity


def LoadCavityFiles():
	Cavity_Checkpoints = [i for i in Path(sys.argv[1]).iterdir() if fnmatch.fnmatch(i, '*CavityProperties*.pk')]
	Time_Snapshots     = []
	Semi_Major_Axis    = []
	Eccentricity       = []
	Argument_Apses     = []
	BinarySMA          = []
	for cc in Cavity_Checkpoints:
		cav_props = load_checkpoint(cc)
        print(cav_props)
		Time_Snapshots.append(cav_props["CurrentTime"])
		Semi_Major_Axis.append(cav_props["SemiMajorAxis"])
		Eccentricity.append(cav_props["Eccentricity"])
		Argument_Apses.append(cav_props["Cavity_Slope_Radians"])
		BinarySMA.append(cav_props["Binary_SemiMajorAxis"])

	lists = list(zip(Time_Snapshots, Semi_Major_Axis, Eccentricity, Argument_Apses,BinarySMA))
	sorted_lists = sorted(lists, key=lambda x: x[0])
	return zip(*sorted_lists)
	 

if __name__ == "__main__":
	
	Cavity_File_Check = CheckForCavityFileExistence()
	if Cavity_File_Check != True:
		print('There are missing cavity property files. Now running fits for',Cavity_File_Check)

		num_tasks_per_batch = 4
		for i in range(0, len(Cavity_File_Check), num_tasks_per_batch):
			with Pool(processes=num_tasks_per_batch) as pool:
				batch       = Cavity_File_Check[i:i + num_tasks_per_batch]
				CavityState = pool.map(MP_Cavity_Properties, batch)

	
	
	sorted_times, sorted_SMA, sorted_ecc, sorted_Apses, sorted_Binary_SMA = LoadCavityFiles()

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
	FigDirectory =  sys.argv[2]
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
