clibs=libvc1_gcc.so libvc2_gcc_O2.so libvc2_gcc_O3.so libvc2_gcc_Ofast.so \
			libvc2_gcc_Ofast_autopar.so libvc3_gcc.so libvc4_gcc.so \
			libvc5_gcc_O2.so libvc5_gcc_O3.so libvc5_gcc_Ofast.so
fortranlibs=vfortran1 vfortran2 vfortran3
libs=$(clibs) $(fortranlibs) 

all: $(libs)

libvc1_gcc.so:
	gcc -shared -march=native -O3 vc1.c -o $@

libvc2_gcc_O2.so:
	gcc -shared -march=native -O2 vc2.c -o $@

libvc2_gcc_O3.so:
	gcc -shared -march=native -O3 vc2.c -o $@
	
libvc2_gcc_Ofast.so:
	gcc -shared -march=native -Ofast vc2.c -o $@

libvc2_gcc_Ofast_autopar.so:
	gcc -shared -march=native -Ofast -floop-parallelize-all -ftree-parallelize-loops=4 vc2.c -o $@
	
libvc3_gcc.so:
	gcc -shared -fopenmp -march=native -O3 vc3.c -o $@

libvc4_gcc.so:
	gcc -shared -fopenmp -march=native -O3 vc4.c -o $@

libvc5_gcc_O2.so:
	gcc -shared -fopenmp -march=native -O2 vc5.c -o $@

libvc5_gcc_O3.so:
	gcc -shared -fopenmp -march=native -O3 vc5.c -o $@

libvc5_gcc_Ofast.so:
	gcc -shared -fopenmp -march=native -Ofast vc5.c -o $@

vfortran1:
	f2py -c vfortran1.f90 -m vfortran1

vfortran2:
	f2py --f90flags="-fopenmp" -c vfortran2.f90 -m vfortran2

vfortran3:
	f2py --f90flags="-fopenmp" -c vfortran3.f90 -m vfortran3

clean:
	rm -f *.so

.phony: clean $(fortranlibs)
