import numpy as np
import math as m
from numpy import array
from numpy import hstack
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns


end_x = 8 # this represents the amount of nodes (notice this is zero based)
end_r = 8 # 

delta_t = 0.0001 #seconds
total_time = 18 #seconds
time = np.linspace(0,total_time,total_time*int(1/delta_t))

#The ambient temperature, ambient pressure and the airspeed around the cylinder
Tambient = (((-7.0263E-17)*(time**6))+((3.7942E-13)*(time**5))-((6.5001E-10)*(time**4))+((2.9117E-7)*time**3)+((2.4549E-4)*(time**2))-((2.4975E-1)*time) +1.4973E1)
Pambient = (((7.8189E-14)*(time**6))-(4.1143E-10)*(time**5)+(9.9884E-7)*(time**4)-(1.3741E-3)*(time**3)+(1.078*(time**2))-(4.3423E2)*time+1.0065E5)
pambient_atm = Pambient/101325
Airspeed = (((3.5132E-16)*(time**6))-((1.8971E-12)*(time**5))+((3.2501E-9)*(time**4))-((1.4559E-6)*(time**3))-((1.2275E-3)*(time**2))+(((1.2487)*(time)))+(1.4909E-1))
#Airspeed = 0*time
# I will define 3 x 3 nodes. there will be symmetry around the axis
# For an element:

#generic values
k = 0.5 #(W/m.K)
rho = 8800 # (kg/m^3)
# Assume constant Cp
Cp = 0.42 #(kJ/kg.K)
D = 0.050 #m

delta_r = (25/end_r)/1000 #m
delta_x = (100/end_x)/1000 #m

# volume element notation as follows: (in m^3)
# first number indicates radial position (inside 1st)
# Second number indicates position along x-axis (inside 1st)

def find_volume_of_side_element(r):
    volume = m.pi*((r*delta_r + delta_r/2)**2 - ((r-1)*delta_r + delta_r/2)**2)*(delta_x/2)
    return volume

def find_volume_centre_element(r):
    volume= m.pi*((r*delta_r + delta_r/2)**2 - ((r-1)*delta_r + delta_r/2)**2)*(delta_x)
    return volume

def find_side_area(r):
    area = m.pi*(((r*delta_r + (delta_r/2))**2)-((r-1)*delta_r + delta_r/2)**2) #m^2
    return area

def find_outer_area_side_element(r):
    area = 2*m.pi*(r*delta_r + (delta_r/2))*(delta_x/2)
    return area

def find_outer_area_middle_element(r):
    area = 2*m.pi*(r*delta_r + (delta_r/2))*(delta_x)
    return area

def find_inner_area_side_element(r):
    area = 2*m.pi*(r*delta_r - (delta_r/2))*(delta_x/2)
    return area

def find_inner_area_middle_element(r):
    area = 2*m.pi*(r*delta_r - (delta_r/2))*(delta_x)
    return area


#def find_volume_top_side_element(r):
#    volume = m.pi*((end_r*delta_r)**2 - (delta_r + delta_r/2)**2)*(delta_x/2)
#    return volume

##def find_volume_top_middle_element(r):
#    volume = m.pi*((end_r*delta_r)**2 - (delta_r + delta_r/2)**2)*(delta_x)
#    return volume 

volume_element_0 = m.pi*((delta_r/2)**2)*(delta_x/2)
volume_element_1 = m.pi*((delta_r/2)**2)*(delta_x)
volume_element_2 = volume_element_0

#volume_element_3 = m.pi*((delta_r + delta_r/2)**2 - (delta_r/2)**2)*(delta_x/2)
#volume_element_4 = m.pi*((delta_r + delta_r/2)**2 - (delta_r/2)**2)*(delta_x)
#volume_element_5 = volume_element_3

volume_element_6 = m.pi*((25/1000)**2 - (25/1000 - delta_r/2)**2)*(delta_x/2)
volume_element_7 = m.pi*((25/1000)**2 - (25/1000 - delta_r/2)**2)*(delta_x)
volume_element_8 = volume_element_6


#volume_total = volume_element_0 + volume_element_1 + volume_element_2 + 2*volume_element_3 + 2*volume_element_4 + 2*volume_element_5 + volume_element_6 + volume_element_7 + volume_element_8
#print("TOTAL VOLUME", volume_total)

#print("volume col 1", volume_element_0 + 2*volume_element_3 + volume_element_6)
#print("volume col 2", volume_element_1 + 2*volume_element_4+ volume_element_7)
#print("volume col 3", volume_element_2+ 2*volume_element_5 +volume_element_8)


#print("volume row 1", volume_element_0 + volume_element_1*99 + volume_element_2)
#print("volume row 2", volume_element_3*99 + volume_element_4*99*99 + volume_element_5*99)
#print("volume row 3", volume_element_6+ volume_element_7*99 +volume_element_8)


#Area of each element at each revevant side:
A_element_0_outer = 2*m.pi*(delta_r/2)*(delta_x/2) #m^2
A_element_0_side = (m.pi*(delta_r/2)**2) #m^2

A_element_1_outer = 2*m.pi*(delta_r/2)*delta_x #m^2
A_element_1_side = (m.pi*(delta_r/2)**2) #m^2

A_element_2_outer = 2*m.pi*(delta_r/2)*(delta_x/2) #m^2
A_element_2_side = (m.pi*(delta_r/2)**2) #m^2

#A_element_3_outer = 2*m.pi*(delta_r + (delta_r/2))*(delta_x/2) #m^2
#A_element_3_inner = 2*m.pi*(delta_r/2)*(delta_x/2)#m^2
#A_element_3_side = m.pi*(((delta_r + (delta_r/2))**2)-((delta_r/2)**2)) #m^2

#A_element_4_outer = 2*m.pi*(delta_r + (delta_r/2))*delta_x #m^2
#A_element_4_inner = 2*m.pi*(delta_r/2)*delta_x #m^2
#A_element_4_side = m.pi*(((delta_r + (delta_r/2))**2)-((delta_r/2)**2)) #m^2

#A_element_5_outer = 2*m.pi*(delta_r + (delta_r/2))*(delta_x/2) #m^2
#A_element_5_inner = 2*m.pi*(delta_r/2)*(delta_x/2) #m^2
#A_element_5_side = m.pi*(((delta_r + (delta_r/2))**2) - ((delta_r/2)**2)) #m^2

A_element_6_outer = 2*m.pi*(end_r*delta_r)*(delta_x/2) + m.pi*((end_r*delta_r)**2 - (end_r*delta_r - delta_r/2)**2)#m^2
A_element_6_inner = 2*m.pi*(end_r*delta_r - delta_r/2)*(delta_x/2) #m^2
A_element_6_side = m.pi*((end_r*delta_r)**2 - (end_r*delta_r - delta_r/2)**2) #m^2

A_element_7_outer = 2*m.pi*(end_r*delta_r)*(delta_x) #m^2
A_element_7_inner = 2*m.pi*(end_r*delta_r - delta_r/2)*(delta_x) #m^2
A_element_7_side =  m.pi*((end_r*delta_r)**2 - (end_r*delta_r - delta_r/2)**2) #m^2

A_element_8_outer = 2*m.pi*(end_r*delta_r)*(delta_x/2) + m.pi*((end_r*delta_r)**2 - (end_r*delta_r - delta_r/2)**2)#m^2
A_element_8_inner = 2*m.pi*(end_r*delta_r - delta_r/2)*(delta_x/2) #m^2
A_element_8_side = m.pi*((end_r*delta_r)**2 - (end_r*delta_r - delta_r/2)**2) #m^2


#outer_area = A_element_0_side + A_element_2_side + A_element_3_side + A_element_5_side + A_element_6_outer + A_element_7_outer + A_element_8_outer
#print(outer_area)

#watt per m^3 is required 
thermal_wattage = 1000*0
e_gen = thermal_wattage/((m.pi)*((25/1000)**2)*(100/1000)) # (W/m^3)
print(e_gen)

#The properties of air will be defined from -50deg Celcius to 100deg Celcius
temperature = array([-50,-40,-30,-20,-10,0,5,10,15,20])  # degrees Celcius
Pr_array = array([0.7440, 0.7436, 0.7425,0.7408, 0.7387, 0.7362,0.7350, 0.7336,0.7323, 0.7309])  # dimensionless 
k_v_array = array(([9.319, 10.08, 10.87, 11.69, 12.52, 13.38, 13.82, 14.27,14.70, 15.16]))*10**-6  # m^2/s
k_array = array([0.01979, 0.02057, 0.02134, 0.02211, 0.02288, 0.02364,0.0241 ,0.02439,0.02476, 0.02514])  # W/m.K

#now we should correct the kinematic viscosity for pressure.
#k_v = k_v[temperature]/(pambient_atm[seconds]**2) #t
##fit a line for each of the air properties so that they can be used for intermediate temperatures
temperature_range = np.linspace(-50, 20, 10)

Pr_m, Pr_b = np.polyfit(temperature, Pr_array, 1)
k_v_m, k_v_b = np.polyfit(temperature, k_v_array, 1)
k_m, k_b = np.polyfit(temperature, k_array, 1)

def find_property (m_value, b_value, temperature_value):
    value = m_value*temperature_value + b_value
    return value

def h (timestamp, temperature):
    #print(timestamp, temperature)
    k_v = find_property(k_v_m, k_v_b, temperature)/((pambient_atm[timestamp])**2)
    Reynolds_number = Airspeed[timestamp]*D/k_v
    k_air = find_property(k_m, k_b, temperature)
    Pr = find_property(Pr_m, Pr_b, temperature) 
    h = (k_air/D)*(0.3 + ((0.62*(Reynolds_number**(1/2))*(Pr**(1/3)))/(1+(0.4/Pr)**(2/3))**(1/4))*(1+(Reynolds_number/282000)**(5/8))**(4/5))
    return h

def average_film_temperature(Node_temperature, timestamp):
    #print(Tambient[timestamp],Node_temperature)
    T_film = (Tambient[timestamp] + Node_temperature)/2
    return T_film


T0_array = []
T2_array = []
T6_array = []
T8_array = []

T = np.full((end_x+1,end_r+1), Tambient[0])
T_new = np.full((end_x+1,end_r+1), Tambient[0])
max_value = Tambient[0]
max_value_index = 0

for x in range(1, end_x):
    globals()['T1_array%s' % x] = []
    globals()['T7_array%s' % x] = []

for r in range(1, end_r):
    globals()['T3_array%s' % r] = []
    globals()['T5_array%s' % r] = []

for x in range(1, end_x):
    for r in range(1, end_r):
        globals()['T4_array%s_%s' %(x, r)] = []

total_volume = 0
for i in range(total_time*int(1/delta_t)): #these are 1/15000 s timesteps


    #Temperature of node 00
    #dubbed element 0
    Tfilm = average_film_temperature(T[0][0], i)
    T0_0_plus_1 = (delta_t/(Cp*rho*volume_element_0))*( 
        (((k*A_element_0_outer*(T[0][1] - T[0][0]))/delta_r)) + 
        (((k*A_element_0_side*(T[1][0] - T[0][0]))/delta_x)) +
        (h(i, Tfilm))*A_element_0_side*(Tambient[i]-T[0][0]) + 
        e_gen*volume_element_0) + T[0][0]
    T_new[0][0] = T0_0_plus_1
    T0_array.append(T0_0_plus_1)
    total_volume = total_volume + volume_element_0
    #print(T0_0_plus_1)

    #Temperature of node 01 (one in, centre bar) T[x][r]
    #dubbed element 1
    for x in range(1, end_x):
        Tx_0_plus_1 = (delta_t/(Cp*rho*volume_element_1))*( 
            (((k*A_element_1_outer*(T[x][1] - T[x][0]))/delta_r)) + 
            (((k*A_element_1_side*(T[x+1][0] - T[x][0]))/delta_x)) +
            (((k*A_element_1_side*(T[x-1][0] - T[x][0]))/delta_x)) +
            e_gen*volume_element_1) + T[x][0]
        T_new[x][0]= Tx_0_plus_1
        globals()['T1_array%s' % x].append(Tx_0_plus_1)
        total_volume = total_volume + volume_element_1


    #Temperature of node 02 [m,n] T[x][r] outer surface, inner bar
    #dubbed element 2
    Tfilm = average_film_temperature(T[end_x][0], i)
    T5000_0_plus_1 =( delta_t/(Cp*rho*volume_element_2))*( 
        (((k*A_element_2_outer*(T[end_x][1] - T[end_x][0]))/delta_r)) + 
        (((k*A_element_2_side*(T[end_x-1][1] - T[end_x][0]))/delta_x)) +
        (h(i, Tfilm))*A_element_2_side*(Tambient[i]-T[end_x][0]) +
        e_gen*volume_element_2) + T[end_x][0]
    T_new[end_x][0] = T5000_0_plus_1
    #print(T5000_0_plus_1)
    T2_array.append(T5000_0_plus_1)
    total_volume = total_volume + volume_element_2

    #Temperature of node 10 [M,N] T[X][R] one up, closed side
    #dubbed element 3
    for r in range(1,end_r):
        Tfilm = average_film_temperature(T[0][r], i)
        T0_r_plus_1 = (delta_t/(Cp*rho*find_volume_of_side_element(r)))*( 
            (((k*find_outer_area_side_element(r)*(T[0][r+1] - T[0][r]))/delta_r)) +
            (((k*find_inner_area_side_element(r)*(T[0][r-1] - T[0][r]))/delta_r)) +  
            (((k*find_side_area(r)*(T[1][r] - T[0][r]))/delta_x)) +
            (h(i, Tfilm))*find_side_area(r)*(Tambient[i]-T[0][r]) +
            e_gen*find_volume_of_side_element(r)) + T[0][r]
        globals()['T3_array%s' % r].append(T0_r_plus_1)
        T_new[0][r] = T0_r_plus_1
        total_volume = total_volume + find_volume_of_side_element(r)
        #print(T0_r_plus_1)

    #Temperature of node 11 [M,N] T[X][R] one up, side
    #dubbed element 4
    for x in range(1, end_x):
        for r in range(1, end_r):
            Tcentre_plus_1 = (delta_t/(Cp*rho*find_volume_centre_element(r)))*( 
                (((k*find_outer_area_middle_element(r)*(T[x][r+1] - T[x][r]))/delta_r)) +
                (((k*find_inner_area_middle_element(r)*(T[x][r-1] - T[x][r]))/delta_r)) +  
                (((k*find_side_area(r)*(T[x-1][r] - T[x][r]))/delta_x)) +
                (((k*find_side_area(r)*(T[x+1][r] - T[x][r]))/delta_x)) +
                e_gen*find_volume_centre_element(r)) + T[x][r]
            globals()['T4_array%s_%s' %(x, r)].append(Tcentre_plus_1)
            T_new[x][r] = Tcentre_plus_1
            total_volume = total_volume + find_volume_centre_element(r)


    #Temperature of node 12 one-up, exposed side [M,N] T[X,R]
    #dubbed element 5
    for r in range(1,end_r):
        Tfilm = average_film_temperature(T[end_x][r], i)
        T5000_r_plus_1 = (delta_t/(Cp*rho*find_volume_of_side_element(r)))*( 
            (((k*find_outer_area_side_element(r)*(T[end_x][r+1] - T[end_x][r]))/delta_r)) +
            (((k*find_inner_area_side_element(r)*(T[end_x][r-1] - T[end_x][r]))/delta_r)) +  
            (((k*find_side_area(r)*(T[end_x-1][r] - T[end_x][r]))/delta_x)) +
            (h(i, Tfilm))*find_side_area(r)*(Tambient[i]-T[end_x][r]) +
            e_gen*find_volume_of_side_element(r)) + T[end_x][r]
        globals()['T5_array%s' % r].append(T5000_r_plus_1)
        T_new[end_x][r] = T5000_r_plus_1
        total_volume = total_volume + find_volume_of_side_element(r)
        #print(T5000_r_plus_1)
    

    #Temperature of node 20
    #element 6
    Tfilm = average_film_temperature(T[0][end_r], i)
    T0_2500_plus_1 =( delta_t/(Cp*rho*volume_element_6))*( 
        (((k*A_element_6_inner*(T[0][end_r-1] - T[0][end_r]))/delta_r)) +  
        (((k*A_element_6_side*(T[1][end_r] - T[0][end_r]))/delta_x)) +
        (h(i, Tfilm))*A_element_6_outer*(Tambient[i]-T[0][end_r]) +
        e_gen*volume_element_6) + T[0][end_r]
    T6_array.append(T0_2500_plus_1)
    T_new[0][end_r] = T0_2500_plus_1
    total_volume = total_volume + volume_element_6
    #print(T0_2500_plus_1)

    #Temperature of node 21
    #element 7
    for x in range(1, end_x):
        Tfilm = average_film_temperature(T[x][end_r], i)
        Tx_2500_plus_1 = (delta_t/(Cp*rho*volume_element_7))*( 
            (((k*A_element_7_inner*(T[x][end_r-1] - T[x][end_r]))/delta_r)) +  
            (((k*A_element_7_side*(T[x+1][end_r] - T[x][end_r]))/delta_x)) +
            (((k*A_element_7_side*(T[x-1][end_r] - T[x][end_r]))/delta_x)) +
            (h(i, Tfilm))*A_element_7_outer*(Tambient[i]-T[x][end_r]) +
            e_gen*volume_element_7) + T[x][end_r]
        T_new[x][end_r] = Tx_2500_plus_1
        globals()['T7_array%s' % x].append(Tx_2500_plus_1)
        total_volume = total_volume + volume_element_7
        #print(Tx_2500_plus_1)

    #Temperature of node 22
    #element 8 
    Tfilm = average_film_temperature(T[end_x][end_r], i)
    T5000_2500_plus_1 = (delta_t/(Cp*rho*volume_element_8))*( 
        (((k*A_element_8_inner*(T[end_x][end_r-1] - T[end_x][end_r]))/delta_r)) +  
        (((k*A_element_8_side*(T[end_x-1][end_r] - T[end_x][end_r]))/delta_x)) +
        (h(i, Tfilm))*A_element_8_outer*(Tambient[i]-T[end_x][end_r]) +
        e_gen*volume_element_8) + T[end_x][end_r]
    T8_array.append(T5000_2500_plus_1)
    T_new[end_x][end_r] = T5000_2500_plus_1
    total_volume = total_volume + volume_element_8

    T = T_new.copy()

    if i == 0:
        print("TOTAL FUCKING VOLUME YOU CUNT",total_volume)

    #This is the section where maximum value is determined. The matrix is flattened
    temporary_max_value = np.amax(T)

    if(temporary_max_value>=max_value):
        max_value = temporary_max_value
        result = np.where(T == temporary_max_value)
        max_value_timestamp = timestamp = i * (1/delta_t)
        Temp_distribution_at_critical = T.copy()

    #print(round(100*(i/(total_time*int(1/delta_t))), 3), "%")


            


print("Maximum value", max_value, "Index" , max_value_index, "coordinates", result, "timestamp", max_value_timestamp)

print(result[0][0], result[0][0])
if result[1][0] == 0:
    if result[0][0] == 0: 
        critical_node_temps = T0_array
    if result[0][0] in range(1,end_x):
        critical_node_temps = globals()['T1_array%s' % str(result[0][0])]
    if result[0][0] == end_x:
        critical_node_temps = T2_array  

if result[1][0] in range(1,end_r):
    if result[0][0] == 0: 
        critical_node_temps = globals()['T3_array%s' % str(result[1][0])]
    if result[0][0] in range(1,end_x):
        critical_node_temps = globals()['T4_array%s_%s' %(str(result[0][0]), result[1][0])]
    if result[0][0] == end_r:
        critical_node_temps = globals()['T5_array%s' % str(result[1][0])]

if result[1][0] == end_r:
    if result[0][0] == 0: 
        critical_node_temps = T6_array
    if result[0][0] in range(1,end_x):
        critical_node_temps = globals()['T7_array%s' % str(result[0][0])]
    if result[0][0] == end_x:
        critical_node_temps = T8_array  

#now I need to find this node in the 2d array and plot it's temperature over time :)

plt.plot(critical_node_temps, label = "Critical node [%s, %s]" %(str(result[0][0]), str(result[1][0])))
plt.legend()
plt.show()

#plt.plot(globals()['T1_array%s' % str(4)], label = "inner node")
#plt.plot(T8_array, label ="outer node")
#plt.legend()
#plt.show()


p1 = sns.heatmap(T_new, square=True,robust=True)
plt.show()

p1 = sns.heatmap(Temp_distribution_at_critical, square=True)
plt.show()

