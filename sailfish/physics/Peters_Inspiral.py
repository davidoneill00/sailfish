from scipy.optimize import newton
import numpy as np

class Orbital_Inspiral():
    r"""
    A class to integrate the orbital elements of a binary undergoing gravitational 
    wave inspiral. 
    """

    def __init__(self,
        GM,
        mass_ratio,
        speed_of_light,
        init_eccentricity,
        init_semimajoraxis,
        timestep,
        plot_inspiral = False
        ): 
        
        def SemiMajorAxis_Decay_Rate(a,e):
            a_dot_prefactor     = 64. / 5. * GM**3 * mass_ratio / (1 + mass_ratio)**2 / speed_of_light**5
            eccentricity_factor = 1 + 73/24 * e**2 + 37/96 * e**4
            return -a_dot_prefactor * eccentricity_factor / (a **3) / ((1-e**2) ** (7./2.)) #* 10
    	
        def Eccentricity_Decay_Rate(a,e):
       	    e_dot_prefactor     = 304. /15. * GM**3 * mass_ratio / (1 + mass_ratio)**2 / speed_of_light**5
            eccentricity_factor = e * (1 + 121/304 * e**2) / ((1-e**2) ** (5./2.))
            return -e_dot_prefactor * eccentricity_factor / a**4  #* 1

        self.a_array    = [init_semimajoraxis]
        self.e_array    = [init_eccentricity]
        self.TimeDomain = [0.]

        # Initialise
        current_time = 0.
        a_old        = init_semimajoraxis
        e_old        = init_eccentricity


        while a_old > 0.03:
            # RK4 integration
            k1_a = SemiMajorAxis_Decay_Rate(a_old , e_old)
            k1_e = Eccentricity_Decay_Rate( a_old , e_old)

            h1_a = timestep * k1_a # step sizes
            h1_e = timestep * k1_e 

            k2_a = SemiMajorAxis_Decay_Rate(a_old + 0.5 * h1_a , e_old + 0.5 * h1_e)
            k2_e = Eccentricity_Decay_Rate( a_old + 0.5 * h1_a , e_old + 0.5 * h1_e)

            h2_a = timestep * k2_a
            h2_e = timestep * k2_e

            k3_a = SemiMajorAxis_Decay_Rate(a_old + 0.5 * h2_a , e_old + 0.5 * h2_e)
            k3_e = Eccentricity_Decay_Rate( a_old + 0.5 * h2_a , e_old + 0.5 * h2_e)

            h3_a = timestep * k3_a
            h3_e = timestep * k3_e

            k4_a = SemiMajorAxis_Decay_Rate(a_old + h3_a , e_old + h3_e)
            k4_e = Eccentricity_Decay_Rate( a_old + h3_a , e_old + h3_e)

            a_new = a_old + (timestep / 6.0) * (k1_a + 2*k2_a + 2*k3_a + k4_a)
            e_new = e_old + (timestep / 6.0) * (k1_e + 2*k2_e + 2*k3_e + k4_e)

            if isinstance(a_new, complex) or isinstance(e_new, complex):
                break
            
            if a_new < 0 or e_new < 0:
                break


            # store values
            self.a_array.append(a_new)
            self.e_array.append(e_new)

            # update for next iteration
            a_old = a_new
            e_old = e_new
            current_time += timestep
            self.TimeDomain.append(current_time)



        if plot_inspiral:
            import matplotlib.pyplot as plt

            Peters_Scale = 4 * 64. / 5. * (GM)**3 * mass_ratio / (1+mass_ratio)**2 / (speed_of_light)**5 / (init_semimajoraxis**4)
            plt.plot(self.TimeDomain,self.a_array,c = 'red',label = 'Semi-Major axis a/a0')
            plt.plot(self.TimeDomain,self.e_array,c = 'blue',label = 'Eccentricity e')
            #plt.plot(self.TimeDomain[0:len(self.a_array)-1],[SemiMajorAxis0 * (1-Peters_Scale* i)**0.25 for i in self.TimeDomain], c = 'black', label = 'Peters, e=0')
            plt.xlabel('Time')
            plt.ylabel('Orbital Elements')
            plt.legend()
            plt.show()
            #plt.savefig('Test.png')
            #import numpy as np
            #print(current_time/2/np.pi)
            #print(1/Peters_Scale/2/np.pi)
            #print(self.TimeDomain[-1])


    

    def f(self,phi, MeanAnomaly, ecc):
        return phi-ecc*np.sin(phi)-MeanAnomaly

    def EccentricAnomaly_from_MeanAnomaly(self,MeanAnomaly,ecc):
        E = newton(self.f, 2,args=(MeanAnomaly,ecc,),tol=0.00001,maxiter=50)
        return E
    
    





#Binary_Orbital_Elements = Orbital_Inspiral(GM = 1,mass_ratio = 1,speed_of_light = 1e1,init_eccentricity= 0.8,init_semimajoraxis= 1, timestep = 1e-1, plot_inspiral = True)
#print(Binary_Orbital_Elements.semimajoraxis)
#print(Binary_Orbital_Elements.eccentricity)


