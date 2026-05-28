OPENQASM 2.0;
include "qelib1.inc";
qreg q94[3];
cx q94[2],q94[1];
rx(3*pi/2) q94[1];
cx q94[0],q94[1];
rx(pi/4) q94[1];
