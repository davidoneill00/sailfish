import sys 
import pickle as pk
import sailfish
import numpy as np 
import matplotlib.pyplot as plt 

Plot = False

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
	
	plt.title('Orbit',chkpt["time"] / 2 / np.pi)
	plt.show()





if __name__ == "__main__":
	from pathlib import Path
	directory = Path(sys.argv[1])
	for arg in directory.iterdir():
		if arg.suffix == '.pk':
			chkpt         = load_checkpoint(arg)
			contour_lines = CavityContour(chkpt)
			print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
			print(arg,'has cavity parameters',MaxDist(contour_lines))
			if Plot:
				main_cbdiso_2d(chkpt,contour_lines)
			else:
				pass
	print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
	exit()

        

