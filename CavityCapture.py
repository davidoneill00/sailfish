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
from sailfish.physics.kepler import OrbitalState

Plot = True

import time

def load_checkpoint(filename):
    with open(filename, "rb") as file:
        chkpt = pk.load(file)
        return chkpt


def CavityContour(chkpt):
	"""
	Extract the density contour points from surface density plot
	"""
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
	try:
		p = plt.contour(f,levels = [0.2],extent=extent).collections[0].get_paths()[0]
	except:
		raise ValueError('chkpt is',chkpt['model_parameters']["nu"],chkpt['time']/2/np.pi)
	v = p.vertices
	v_resampled = scipy.signal.resample(v, 500)
	return v_resampled


def MaxDist(points):
	"""
	Based on surface density contour, fit with an ellipse, with the properties of the ellipse returned as a .pk file
	"""

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



def main_cbdiso_2d(chkpt,points,in_dir,filename):
	"""
	Optional step, save the figure with the density contour overplotted
	"""
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

	#primary, secondary = chkpt["point_masses"]
	#true_anomaly       = OrbitalState(primary, secondary).true_anomaly(chkpt["time"])

	def Parameterisation(a,e,f,w):
		r = a * (1-e**2) / (1+e*np.cos(f-w))
		return apses_x[0] - a*(1-e)*np.cos(w) + r * (np.cos(f)), apses_y[0] -a*(1-e)*np.sin(w) + r * (np.sin(f))
		
	Position_List = np.array([Parameterisation(a_cavity,e_cavity,i,arg_aps) for i in np.linspace(0,2*np.pi,100)]).reshape(100,2)
	plt.plot(Position_List[:,0],Position_List[:,1],c = 'black', linestyle = 'dashed', linewidth = 2)
	
	fig.suptitle('Orbit %g'%(chkpt["time"] / 2 / np.pi))
	FigDirectory =  in_dir
	ax.set_xlim(-4, 4)
	ax.set_ylim(-4, 4)
	plt.savefig(filename.replace(".pk", ".png"), dpi = 400)
	plt.close(fig)


def MP_Cavity_Properties(arg,in_dir):
	"""
	This .pk only knows about the properties of the cavity so far.
	We want to give it information on the orbital state of the system too.
	"""


	chkpt             = load_checkpoint(arg)
	contour_lines     = CavityContour(chkpt)
	cavity_properties = MaxDist(contour_lines)

	Binary_SMA        = np.array([s[ 1] for s in chkpt['timeseries']])[-1]
	cavity_properties["Binary_SemiMajorAxis"] = Binary_SMA
	cavity_properties["CurrentTime"]          = chkpt["time"] / 2 / np.pi
	cavity_properties["viscosity"]            = chkpt["model_parameters"]["nu"]
	cavity_properties["Retrograde"]           = chkpt["model_parameters"]["retrograde"]

	match                             = re.search(r'chkpt\.(\d+)', arg)
	Number                            = match.group(1)
	cavity_properties["ChkptNumbers"] = Number

	point_mass1, point_mass2 = chkpt["point_masses"]
	#TrueAnomaly              = OrbitalState(point_mass1, point_mass2).true_anomaly(cavity_properties["CurrentTime"] * 2 * np.pi)
	TrueAnomaly              = np.arctan2(point_mass1.position_y, point_mass1.position_x)
	CurrentAngle             = np.copy(cavity_properties["Cavity_Slope_Radians"]) #+= -TrueAnomaly
	cavity_properties["Cavity_Slope_Radians"] = CurrentAngle - TrueAnomaly
	if Plot:
		main_cbdiso_2d(chkpt,contour_lines,in_dir,arg)
	return cavity_properties






def FitCavityCheck(in_dir):
	"""
	If rerunning the this file, we want to be able to skip completed checkpoints.
	This function returns 'which_ones' to run as well as the viscosity (for convenience).
	"""

	import re

	directory          = Path(in_dir)
	Checkpoint_Numbers = []
	nu                 = None
	
	for i in os.listdir(directory):
		if fnmatch.fnmatch(i, 'CavityProperties_nu.*.pk'):
			Cavity_Loaded_ = load_checkpoint(in_dir + '/' + i)
		elif fnmatch.fnmatch(i, '*chkpt*.pk'):
			match  = re.search(r'chkpt\.(\d+)', i)
			Number = match.group(1)
			Checkpoint_Numbers.append(Number)

			if nu == None:
				nu = load_checkpoint(in_dir + '/' + i)['model_parameters']['nu']



	try:
		MissingNumbers = list(set(Checkpoint_Numbers) - set(list(Cavity_Loaded_["ChkptNumbers"])))

	except:
		MissingNumbers = Checkpoint_Numbers

	which_ones = [f"chkpt.{int(number):04d}.pk" for number in MissingNumbers]
	return nu, which_ones






def CavityEvolution(in_dir):

	"""
	Given a list of 'which_ones' to run, the cavities are profiled below and appended to the existing (or not)
	cavity properties file. This dictionary then overwrites any pre existing cavity property files
	"""

	print('Looking at directory:',in_dir)
	nu, which_ones = FitCavityCheck(in_dir)
	CavityFileName = in_dir + '/CavityProperties_nu.%g.pk'%(nu)

	print(which_ones)
	try:
		Cavity_Loaded_  = load_checkpoint(CavityFileName)
		Time_Snapshots  = Cavity_Loaded_["Timeseries"]
		Semi_Major_Axis = Cavity_Loaded_["SemiMajor_Axis"]
		Eccentricity    = Cavity_Loaded_["Eccentricity"]
		Argument_Apses  = Cavity_Loaded_["Inclination"]
		BinarySMA       = Cavity_Loaded_["Binary_SMA"]
		ChkptNumbers    = Cavity_Loaded_["ChkptNumbers"]
		print('Found Pre existing File')

	except:
		Time_Snapshots  = []
		Semi_Major_Axis = []
		Eccentricity    = []
		Argument_Apses  = []
		BinarySMA       = []
		ChkptNumbers    = []


	for file in which_ones:
		
		if fnmatch.fnmatch(in_dir + '/' + file, '*chkpt*.pk'):
			cav_props = MP_Cavity_Properties(in_dir + '/' + file, in_dir)
			
			Time_Snapshots.append(cav_props["CurrentTime"])
			Semi_Major_Axis.append(cav_props["SemiMajorAxis"])
			Eccentricity.append(cav_props["Eccentricity"])
			Argument_Apses.append(cav_props["Cavity_Slope_Radians"])
			BinarySMA.append(cav_props["Binary_SemiMajorAxis"])
			match  = re.search(r'chkpt\.(\d+)', file)
			Number = match.group(1)
			ChkptNumbers.append(Number)

			print('Finished Processing checkpoint number',Number)
			
		else:
			pass


	lists         = list(zip(Time_Snapshots, Semi_Major_Axis, Eccentricity, Argument_Apses,BinarySMA,ChkptNumbers))
	sorted_lists  = sorted(lists, key=lambda x: x[0])

	sorted_times, sorted_SMA, sorted_ecc, sorted_Apses, sorted_Binary_SMA, sorted_ChkptNumbers = zip(*sorted_lists)

	if which_ones != []:
		Cavity = {
		'Timeseries':sorted_times,
		'SemiMajor_Axis':sorted_SMA,
		'Eccentricity':sorted_ecc,
		'Inclination':sorted_Apses,
		'Binary_SMA':sorted_Binary_SMA,
		'nu':nu,
		'Retrograde':True,
		'ChkptNumbers':sorted_ChkptNumbers
		}

		with open(CavityFileName, "wb") as cvt:
			pk.dump(Cavity, cvt)

	return CavityFileName




def Plot_Cavitites(Cavity):

	if Cavity['Retrograde'] == False:
		Linestyle = 'dotted'
		Marker    = '*'
		Label1     = 'Prograde '
	elif Cavity['Retrograde'] == True:
		Linestyle = 'solid'
		Marker    = 'o'
		Label1     = 'Retrograde '
	if Cavity['nu'] == 0.01:
		Colour = 'blue'
		Label2 = 'nu=1e-2'
	elif Cavity['nu'] == 0.003:
		Colour = 'orange'
		Label2 = 'nu=3e-3'
	elif Cavity['nu'] == 0.001:
		Colour = 'green'
		Label2 = 'nu=1e-3'
	elif Cavity['nu'] == 0.0003:
		Colour = 'red'
		Label2 = 'nu=3e-4'
	elif Cavity['nu'] == 0.0001:
		Colour = 'purple'
		Label2 = 'nu=1e-4'

	Label = Label1 + Label2

	plt.plot(np.array(Cavity['Binary_SMA']),Cavity['SemiMajor_Axis'],c=Colour, linestyle=Linestyle)
	plt.scatter(np.array(Cavity['Binary_SMA']),Cavity['SemiMajor_Axis'],c=Colour, marker=Marker, s = 50, label = Label)



def Plot_Properties_of_Cavity(Cavity,FigDirectory):

	fig, ax = plt.subplots(figsize=[12, 9])
	if Cavity['Retrograde']:
		plt.title('Cavity Properties Retrograde nu = %g'%(Cavity['nu']))
		pngname = FigDirectory + f"{'/CavityProperties_Prograde_nu'}.{Cavity['nu']}.png"
	else:
		plt.title('Cavity Properties Prograde nu = %g'%(Cavity['nu']))
		pngname = FigDirectory + f"{'/CavityProperties_Retrograde_nu'}.{Cavity['nu']}.png"

	plt.plot(Cavity['Timeseries'],Cavity['SemiMajor_Axis'], c = 'brown',linewidth = 2, label = r'Semi Major Axis $[a_0]$')
	plt.plot(Cavity['Timeseries'],Cavity['Eccentricity'], c = 'blue', linestyle = 'dashed',linewidth = 2, label = 'Eccentricity')
	plt.plot(Cavity['Timeseries'],Cavity['Inclination'], c = 'silver', linestyle = 'dotted',linewidth = 2, label = 'Apsidal Inclination (Radians)')
	plt.scatter(Cavity['Timeseries'],Cavity['SemiMajor_Axis'], c = 'brown', s=50)
	plt.scatter(Cavity['Timeseries'],Cavity['Eccentricity'], c = 'blue', s=50)
	plt.scatter(Cavity['Timeseries'],Cavity['Inclination'], c = 'silver', s=50)
	plt.xlabel(r'Time $2\pi\Omega_0^{-1}$')
	plt.legend()
	plt.xlim([1000,plt.gca().get_xlim()[1]])
	fig.savefig(pngname, dpi=400)
	plt.close(fig)


def ReProcessCavityPickleFile(Cavity,in_dir):
	if excluded_names[Cavity['nu']] !=[]:
		excluded_files = [in_dir + "/" + i for i in excluded_names[Cavity['nu']]]
		print('All files being excluded from nu ',Cavity['nu'],' are: ',excluded_files)
		skip_index     = []
		for i in excluded_files:
			chkpt = load_checkpoint(i)
			skip_index.append(np.where(np.isclose(Cavity['Timeseries'], chkpt["time"]/2/np.pi, atol=0.001))[0][0])

		ProcessedCavity = {}
		ProcessedCavity['nu']         = Cavity['nu']
		ProcessedCavity['Retrograde'] = Cavity['Retrograde']
		for k in Cavity.keys():
			if k!='nu' and k!='Retrograde':
				for ind in skip_index:
					ProcessedCavity[k] = np.delete(Cavity[k], ind)
	else:
		ProcessedCavity = Cavity

	return ProcessedCavity

def Compare_Cavities(parent_dir,thread=None):
	"""
	Final function to run, putting all of the above together. Additional option 'thread' means a number of 
	independent 'threads' can be run simultaneously. The thread number basically just sets the directory which each 
	process will work on individually.
	"""

	subdirs = [parent_dir + '/' + name for name in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, name))]
	fig, ax = plt.subplots(figsize=[12, 9])
	plt.title('Cavity Semi Major Axis',fontsize=25)
	plt.xlabel(r'$\log_{10} a_\mathrm{bin}~[a_0]$',fontsize=20)
	plt.ylabel(r'$\log_{10} a_\mathrm{cav}~[a_0]$',fontsize=20)

	if thread == None:
		Evaluate = subdirs
		print('-----------------------------------------------------')
		print('Running all directories',Evaluate)
		print('-----------------------------------------------------')
	else:
		Evaluate = subdirs[thread:thread+1]
		print('-----------------------------------------------------')
		print('Thread',thread,'Corresponding to subdirectory',Evaluate)
		print('-----------------------------------------------------')

	for in_dir in Evaluate:
		CavityFileName    = CavityEvolution(in_dir)
		Cavity            = load_checkpoint(CavityFileName)
		ReprocessedCavity = ReProcessCavityPickleFile(Cavity,in_dir)
		Plot_Cavitites(ReprocessedCavity)

	plt.legend()
	plt.gca().invert_xaxis()
	ax.set_yscale('log')
	ax.set_xscale('log')
	plt.yticks([10,1,0.1,0.05 ])
	plt.xticks([1,0.1,0.01])
	plt.savefig(parent_dir + '/Decoupling.png', dpi=400)
	plt.close(fig)

	#for in_dir in subdirs:
	#	CavityFileName    = CavityEvolution(in_dir)
	#	Cavity            = load_checkpoint(CavityFileName)
	#	ReprocessedCavity = ReProcessCavityPickleFile(Cavity,in_dir)
	#	Plot_Properties_of_Cavity(ReprocessedCavity,parent_dir)


#Exclusion_List =
#for i in range():
#	'chkpt.0'

excluded_names = {}
excluded_names[0.0001] = []
excluded_names[0.0003] = []
excluded_names[0.001]  = []
excluded_names[0.003]  = []

try:
	Compare_Cavities(parent_dir = sys.argv[1], thread = int(sys.argv[2]))
except:
	Compare_Cavities(parent_dir = sys.argv[1])



exit()
