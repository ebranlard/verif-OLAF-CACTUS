#
Comparison cases between CACTUS and OLAF.
Using personnal branch of CACTUS (with extra flags implemented)


# Cases
- `B1S1_I0_DS0_AM0/` : one blade, one section, no induction, no dynamic stall, no added mass no 
- `B1S1_I1_DS1_AM0/` : one blade, one section, with induction




# NOTES

CACTUS:
- Cactus uses just a lifting line, no TE

- Workflow
    - Convergence loop for bound circulation 
         Trailed circulation is assumed known from previous time step, only the bound circulation is updatd TODO make sure of that
        - UpdateBladeVel: Compute Wake Vel, StrVel and IndVel at Nodes (not CP). 
             First iteration compute the influence of entire wake on LL (without LL) and influence of LL
             Second iterations, only compute of LL (bound circulation has changed, wake influence is the same)
             - BladeIndVel: Compute induced velocity of wake (with or without LL) at given point
                    a flag decide whether to use all the elements)
                    or just the lifting lines (of all blades) (k=nt)
                    or all wake minus bound lifting line (goint to NT1=nt-1, nt is LL)
                  -VorIVel: induced velocity from one segment
                        different cores are used for Bound/ Trailed and Span (VFlag)
        - BladeLoads: Compute loads on all blade elements
            - bsload:
                - Average between nodes 
                - Compute alpha at 1/4 chord (LL)
                - Compute alpha at .05 and 3/4 chord, NOTE: the induction is the same, but only the structural velocity is changed!
                - DynStall
                - Added mass
        -
        -
        - Outputs
        - UpdateWakeVel: Compute induced velocity on all elements
        - Conlp: convect the wake ("upwinded" using old velocity)
        - Optional Filter of bound circulation (instead of iteration) 
        - Update states

- Cactus use Cldyn-> GB_raw -> GammaNearWake


Comparison Cactus
-----------------
>>> Be careful of viscosity value it was wrong in previous input files
>>> Be careful of RE interp, compile openfast with flag
>>> DynaStall and Added mass have an impact
     prflag=1 AddMassFlag=1

Scaling
-------
