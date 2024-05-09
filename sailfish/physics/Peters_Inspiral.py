#from typing import NamedTuple
import numpy as np
import matplotlib.pyplot as plt

class Orbital_Inspiral():
    r"""
    A class to integrate the orbital elements of a binary undergoing gravitational 
    wave inspiral. 
    """

    def __init__(self,
        current_time,
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
            return - beta * eccentricity_factor / a **3 / ((1-e**2) ** 3.5)
    	
        def Eccentricity_Decay_Rate(a,e):
       	    gamma               = 304. /15. * e * GM**3 * mass_ratio / (1 + mass_ratio)**2 / speed_of_light**5
            eccentricity_factor = 1 + 121/304 * e**2
            return -gamma * eccentricity_factor / a**4 / ((1-e**2) ** 2.5)

        self.a_array = []
        self.e_array = []
        self.T       = np.arange(0,current_time+timestep,timestep)

        a_old        = SemiMajorAxis0
        e_old        = eccentricity0

        for i in range(len(self.T)):
            a_new = a_old + Semi_Major_Axis_Decay_Rate(a_old,e_old,) * timestep 
            e_new = e_old + Eccentricity_Decay_Rate(a_old,e_old) * timestep 

            if a_new < 0:
                # Add a merger flag?
                break

            else:
                self.a_array.append(a_new)
                self.e_array.append(e_new)
                a_old = a_new
                e_old = e_new

        self.semimajoraxis = a_old
        self.eccentricity  = e_old
        self.TimeDomain    = self.T[0:len(self.a_array)]

        if plot_inspiral:
            Peters_Scale = 4 * 64. / 5. * (GM)**3 * mass_ratio / (1+mass_ratio)**2 / (speed_of_light)**5 / (SemiMajorAxis0**4)
            plt.plot(self.TimeDomain,self.a_array,c = 'red',label = 'Semi-Major axis a/a0')
            plt.plot(self.TimeDomain,self.e_array,c = 'blue',label = 'Eccentricity e')
            plt.plot(self.TimeDomain,[SemiMajorAxis0 * (1-Peters_Scale* i)**0.25 for i in self.TimeDomain], c = 'black', label = 'Peters, e=0')
            plt.xlabel('Time')
            plt.ylabel('Orbital Elements')
            plt.legend()
            plt.show()





#Binary_Orbital_Elements = Orbital_Inspiral(current_time = 1e5,GM = 1,mass_ratio = 1,speed_of_light = 1e1,eccentricity0 = 0.9,SemiMajorAxis0 = 1, timestep = 1e2, plot_inspiral = True)
#print(Binary_Orbital_Elements.semimajoraxis)
#print(Binary_Orbital_Elements.eccentricity)


