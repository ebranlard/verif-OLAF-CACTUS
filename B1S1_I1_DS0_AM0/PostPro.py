import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import welib.fast.fastlib as fastlib
# Local 
import weio
import f90nml


outputfolder = './'
folder='.'

case='CACTUS'


D     = 2
R     = D/2
A     = D**2    # <<<<<<<<<<<<<< NOTE:
AQ    = R**2    # <<<<<<<<<<<<<< NOTE:
rho   = 1.2
nu    = 1.55e-5

cols=weio.read('./RenamedColumns.csv').toDataFrame().columns.values

res = pd.DataFrame()

files=glob.glob(os.path.join(folder,'output','*TimeData.csv'))
# files=glob.glob(os.path.join(folder,'*.in'))
files.sort()
print(files)

for i in np.arange(len(files)):
    f = files[i]

    # --- Read input file to get operating conditions
    input_file = os.path.join(folder, os.path.basename(f[:-13])+'.in')
    #     input_file = os.path.join(folder, 'XFE_halfScale_{}.in'.format(i+1))
    #     print(input_file)
    nml = f90nml.read(input_file)
    nRot    = nml['configinputs']['nr']
    nPerRot = nml['configinputs']['nti']
    # case
    RPM = nml['caseinputs']['rpm']
    TSR = nml['caseinputs']['ut']
    CTExcrM = nml['caseinputs']['ctexcrm']




    # --- Read data and set sim values
    df = weio.read(f).toDataFrame()
    if len(df.columns)!=len(cols):
        print(df.columns)
        print(len(df.columns))
        print(len(cols))
        raise Exception('Problem with number of columns')
    df.columns=cols

    # --- Read elem
    elemfile = f.replace('TimeData','ElementData')
    dfElem = weio.read(elemfile).toDataFrame()
    print(dfElem.columns)
    print(dfElem.shape)
    print(df.shape)


    timeSteps = np.arange(0,nRot*nPerRot)
    omega     = RPM*2*np.pi/60
    U         = omega*R/TSR
    T         = 2*np.pi/(omega)
    dt        = T/nPerRot
    time      = timeSteps*dt
    theta     = time*omega
    time_norm = df['Normalized Time (-)'].values
    dt_norm   = time_norm[1]-time_norm[0]
    Utip = omega*R

    # --- Create a fake df for comparison
    P0 = 0.5 *rho * A * U**3
    F0 = 0.5 *rho * A * U**2
    Q0 = 0.5 *rho * A * U**2 * R
    Q0e = 0.5 *rho * AQ*R * Utip**2
    ones = np.ones(time.shape)

    if (np.abs(df['Theta (rad)'].values[-1]-theta[-1])>0.0001):
        print('>>> Wrong theta')
    if (df.shape[0]!=len(time)):
        print('>>> Wrong nsteps')
        time  = time[:df.shape[0]]
        theta = theta[:df.shape[0]]
        ones  = ones[:df.shape[0]]

    c=0
    # y CACTUS seems to be pointing up
    df.insert(c , 'Time_[s]'        , time)                                    ; c+=1
    df.insert(c , 'Azimuth_[deg]'   , np.mod(theta*180/np.pi , 360))           ; c+=1
    df.insert(c , 'HWindSpeedX_[m/s]' , U*ones)                                ; c+=1
    df.insert(c , 'RtAeroCp_[-]'    , df['Power Coeff. (-)'])                  ; c+=1
    df.insert(c , 'RtAeroCq_[-]'    , df['Torque Coeff. (-)'])                 ; c+=1
    df.insert(c , 'ZAeroFxg_[N]'   , df['Fx Coeff. (-)']*F0)                  ; c+=1
    df.insert(c , 'ZAeroFyg_[N]'   ,-df['Fz Coeff. (-)']*F0)                  ; c+=1
    df.insert(c , 'ZAeroFzg_[N]'   , df['Fy Coeff. (-)']*F0)                  ; c+=1
    df.insert(c , 'ZAeroMzg_[N-m]' , df['Torque Coeff. (-)']*Q0)              ; c+=1

    # TODO this need cos/sin azimuth
    df.insert(c , 'RtAeroFxh_[N]'   , df['Fy Coeff. (-)']*F0)                  ; c+=1

    df.insert(c , 'RtAeroFyh_[N]'   , (np.cos(theta)*df['Fx Coeff. (-)'] - np.sin(theta)*df['Fz Coeff. (-)'])*F0  ); c+=1
    df.insert(c , 'RtAeroFzh_[N]'   ,-(np.cos(theta)*df['Fz Coeff. (-)'] + np.sin(theta)*df['Fx Coeff. (-)'])*F0  ); c+=1
    df.insert(c , 'RtAeroMxh_[N-m]' , df['Torque Coeff. (-)']*Q0)              ; c+=1
    df.insert(c , 'RtAeroPwr_[W]'   , df['Power Coeff. (-)']*P0)               ; c+=1

    df.insert(c , 'RotSpeed_[rpm]'   , RPM*ones)                                ; c+=1

    psi = theta - 0
    df.insert(c , 'Y1AeroFxb_[N]'   , -(np.cos(psi)*df['Blade1 Fz Coeff. (-)'] + np.sin(psi)*df['Blade1 Fx Coeff. (-)'])*F0)           ; c+=1
    df.insert(c , 'Y1AeroFyb_[N]'   ,  (np.cos(psi)*df['Blade1 Fx Coeff. (-)'] - np.sin(psi)*df['Blade1 Fz Coeff. (-)'])*F0)           ; c+=1
    df.insert(c , 'Y1AeroFzb_[N]'   ,-df['Blade1 Fy Coeff. (-)']*F0)           ; c+=1
    df.insert(c , 'Y1AeroMxb_[N-m]' , df['Blade1 Torque Coeff. (-)']*Q0)       ; c+=1
    df.insert(c , 'Y1AeroPwr_[W]'   , df['Blade1 Torque Coeff. (-)']*Q0*omega) ; c+=1
    df.insert(c , 'Z1AeroFxg_[N]'   , df['Blade1 Fx Coeff. (-)']*F0)           ; c+=1
    df.insert(c , 'Z1AeroFyg_[N]'   ,-df['Blade1 Fz Coeff. (-)']*F0)           ; c+=1
    df.insert(c , 'Z1AeroFzg_[N]'   , df['Blade1 Fy Coeff. (-)']*F0)           ; c+=1
    df.insert(c , 'Z1AeroMzg_[N-m]' , df['Blade1 Torque Coeff. (-)']*Q0)       ; c+=1

    df.insert(c , 'Wind1VelX_[m/s]' , U*ones)                                  ; c+=1


    # ---
    IBld  = np.unique(dfElem['Blade']  ).astype(int)
    for iB in IBld:
        dfBld=dfElem[dfElem['Blade']==iB]
        IElem = np.unique(dfBld['Element']).astype(int)
        for ie in IElem:
            dfSec = dfBld[dfBld['Element']==ie]
            if dfSec.shape[0]!=df.shape[0]:
                print('>>> Inconsistent shape, ',iB,ie)
            else:
                # TODO x/y/R
                df.insert(c , 'AB{:d}N{:03d}Alpha_[deg]'.format(iB,ie)   , -dfSec['AOA25 (deg)'].values); c+=1
                df.insert(c , 'AB{:d}N{:03d}Alpha50_[deg]'.format(iB,ie) , -dfSec['AOA50 (deg)'].values); c+=1
                df.insert(c , 'AB{:d}N{:03d}Alpha75_[deg]'.format(iB,ie) , -dfSec['AOA75 (deg)'].values); c+=1
                df.insert(c , 'AB{:d}N{:03d}AlphaDot_[-]'.format(iB,ie)  , -dfSec['AdotNorm (-)'].values); c+=1 # TODO
                df.insert(c , 'AB{:d}N{:03d}Cl_[-]'     .format(iB,ie)   , -dfSec['CL (-)'].values); c+=1
                df.insert(c , 'AB{:d}N{:03d}Cd_[-]'     .format(iB,ie)   ,  dfSec['CD (-)'].values); c+=1
                df.insert(c , 'AB{:d}N{:03d}Cm_[-]'     .format(iB,ie)   , -dfSec['CM25 (-)'].values); c+=1
                df.insert(c , 'AB{:d}N{:03d}Cn_[-]'     .format(iB,ie)   , -dfSec['CN (-)'].values); c+=1
                df.insert(c , 'AB{:d}N{:03d}Ct_[-]'     .format(iB,ie)   , -dfSec['CT (-)'].values); c+=1
                df.insert(c , 'AB{:d}N{:03d}Cxg_[-]'     .format(iB,ie)  ,  dfSec['Fx (-)'].values); c+=1 # TODO, this is likely coefficients related to global coords
                df.insert(c , 'AB{:d}N{:03d}Cyg_[-]'     .format(iB,ie)  , -dfSec['Fz (-)'].values); c+=1 # TODO
                df.insert(c , 'AB{:d}N{:03d}Czg_[-]'     .format(iB,ie)  ,  dfSec['Fy (-)'].values); c+=1 # TODO
                df.insert(c , 'AB{:d}N{:03d}ClC_[-]'    .format(iB,ie)   , -dfSec['CLCirc (-)'].values); c+=1
                df.insert(c , 'AB{:d}N{:03d}Re_[-]'     .format(iB,ie)   ,  dfSec['Re (-)'].values/1e6); c+=1
                df.insert(c , 'AB{:d}N{:03d}Gam_[m^2/s]'.format(iB,ie)   , -dfSec['GB (?)'].values*U*R); c+=1 # TODO
                df.insert(c , 'AB{:d}N{:03d}Vrel_[m/s]' .format(iB,ie)   ,  dfSec['Ur (-)'].values*U); c+=1
                df.insert(c , 'AB{:d}N{:03d}Vindx_[m/s]'.format(iB,ie)   ,  dfSec['IndU (-)'].values*U); c+=1 # TODO
                df.insert(c , 'AB{:d}N{:03d}Vindy_[m/s]'.format(iB,ie)   ,  dfSec['IndV (-)'].values*U); c+=1 # TODO
                df.insert(c , 'AB{:d}N{:03d}Vindz_[m/s]'.format(iB,ie)   ,  dfSec['IndW (-)'].values*U); c+=1 # TODO


    #df.insert(c , 'Y1AeroPwr_[W]'   , df['Blade1 Torque Coeff. (-)']*Q0*omega) ; c+=1
    #df.insert(c , 'Y2AeroPwr_[W]'   , df['Blade2 Torque Coeff. (-)']*Q0*omega) ; c+=1
    #df.insert(c , 'Y3AeroPwr_[W]'   , df['Blade3 Torque Coeff. (-)']*Q0*omega) ; c+=1
    #df.insert(c , 'Y4AeroPwr_[W]'   , df['Blade3 Torque Coeff. (-)']*Q0*omega) ; c+=1
#     df.insert(c , 'Y5AeroPwr_[W]'   , df['Blade2 Torque Coeff. (-)']*Q0*omega) ; c+=1
#     df.insert(c , 'Y6AeroPwr_[W]'   , df['Blade9 Torque Coeff. (-)']*Q0*omega) ; c+=1
#     df.insert(c , 'Y7AeroPwr_[W]'   , df['Blade8 Torque Coeff. (-)']*Q0*omega) ; c+=1
#     df.insert(c , 'Y8AeroPwr_[W]'   , df['Blade6 Torque Coeff. (-)']*Q0*omega) ; c+=1
#     df.insert(c , 'Y9AeroPwr_[W]'   , df['Blade5 Torque Coeff. (-)']*Q0*omega) ; c+=1



    df['RtAeroPwr_[W]'].values[0]=df['RtAeroPwr_[W]'].values[1]
    df['Y1AeroPwr_[W]'].values[0]=df['Y1AeroPwr_[W]'].values[1]
#     df['Y4AeroPwr_[W]'].values[0]=df['Y4AeroPwr_[W]'].values[1]
#     df['Y5AeroPwr_[W]'].values[0]=df['Y5AeroPwr_[W]'].values[1]
#     df['Y6AeroPwr_[W]'].values[0]=df['Y6AeroPwr_[W]'].values[1]
#     df['Y7AeroPwr_[W]'].values[0]=df['Y7AeroPwr_[W]'].values[1]
#     df['Y8AeroPwr_[W]'].values[0]=df['Y8AeroPwr_[W]'].values[1]
#     df['Y9AeroPwr_[W]'].values[0]=df['Y9AeroPwr_[W]'].values[1]
# 

    outputfile = os.path.join(outputfolder, 'CACTUS.csv'.format(case,TSR))
    df.to_csv(outputfile, index=False)

