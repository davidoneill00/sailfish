#from typing import NamedTuple
import numpy as np
import matplotlib.pyplot as plt

class Orbital_Inspiral():
    r"""
    A class to integrate the orbital elements of a binary undergoing gravitational 
    wave inspiral. 
    """

    def __init__(self,
        GM,
        mass_ratio,
        speed_of_light,
        eccentricity0,
        SemiMajorAxis0,
        timestep,
        plot_inspiral = False
        ): 
        
        def Semi_Major_Axis_Decay_Rate(a,e):
            beta                = 64. / 5. * GM**3 * mass_ratio / (1 + mass_ratio)**2 / speed_of_light**5
            eccentricity_factor = 1 + 73/24 * e**2 + 37/96 * e**4
            return - beta * eccentricity_factor / a **3 / ((1-e**2) ** 3.5) #* 10
    	
        def Eccentricity_Decay_Rate(a,e):
       	    gamma               = 304. /15. * e * GM**3 * mass_ratio / (1 + mass_ratio)**2 / speed_of_light**5
            eccentricity_factor = 1 + 121/304 * e**2
            return -gamma * eccentricity_factor / a**4 / ((1-e**2) ** 2.5) #* 1

        def Circular_Inspiral_Time():
            a0   = 1.
            beta = 64. / 5. * GM**3 * mass_ratio / (1 + mass_ratio)**2 / speed_of_light**5
            return a0**4 / (4. * beta)

        self.a_array = [SemiMajorAxis0]
        self.e_array = [eccentricity0]
        self.T       = np.arange(0,Circular_Inspiral_Time(),timestep)

        a_old        = SemiMajorAxis0
        e_old        = eccentricity0

        for i in range(1,len(self.T)):
            a_new = a_old + Semi_Major_Axis_Decay_Rate(a_old,e_old) * timestep 
            e_new = e_old + Eccentricity_Decay_Rate(a_old,e_old) * timestep 

            if a_new < 0:
                a_new = 1e-5
                e_new = 0.
                self.a_array.append(a_new)
                self.e_array.append(e_new)
                # Merger has occured. We fix a small semi-major axis to avoid
                # divergences of an a = 0 binary
                break

            else:
                self.a_array.append(a_new)
                self.e_array.append(e_new)
                a_old = a_new
                e_old = e_new

        self.TimeDomain    = self.T[0:len(self.a_array)]

        if plot_inspiral:
            Peters_Scale = 4 * 64. / 5. * (GM)**3 * mass_ratio / (1+mass_ratio)**2 / (speed_of_light)**5 / (SemiMajorAxis0**4)
            plt.plot(self.TimeDomain,self.a_array,c = 'red',label = 'Semi-Major axis a/a0')
            plt.plot(self.TimeDomain,self.e_array,c = 'blue',label = 'Eccentricity e')
            #plt.plot(self.TimeDomain[0:len(self.a_array)-1],[SemiMajorAxis0 * (1-Peters_Scale* i)**0.25 for i in self.TimeDomain], c = 'black', label = 'Peters, e=0')
            plt.xlabel('Time')
            plt.ylabel('Orbital Elements')
            plt.legend()
            plt.show()





#Binary_Orbital_Elements = Orbital_Inspiral(GM = 1,mass_ratio = 1,speed_of_light = 1e1,eccentricity0 = 0,SemiMajorAxis0 = 1, timestep = 1e-3, plot_inspiral = True)
#print(Binary_Orbital_Elements.semimajoraxis)
#print(Binary_Orbital_Elements.eccentricity)


