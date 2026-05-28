OPENQASM 2.0;
include "qelib1.inc";
qreg q578[3];
cx q578[0],q578[1];
rx(3*pi/4) q578[2];
cx q578[1],q578[2];
cx q578[0],q578[1];
