

all: cc co rc ro

# all: cc rc2

cc:
	cd ../../CACTUS/build/ && make 

co:
	cd ../../openfast/build-single-debug/ && make  aerodyn_driver

rc:
	del /S /Q output
	..\..\CACTUS\bin\cactus.exe CACTUS.in  > OUT_CACT
	python .\PostPro.py

rc2:
	del /S /Q output
	..\..\CACTUS\bin\cactus.exe CACTUS.in 
	python .\PostPro.py

ro:
	del /S /Q *.outb
	..\..\openfast\build-single-debug\modules\aerodyn\aerodyn_driver.exe OLAF.dvr > OUT_OLAF
