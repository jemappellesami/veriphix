OPENQASM 2.0;
include "qelib1.inc";
qreg q600[3];
cx q600[1],q600[0];
cx q600[2],q600[1];
rx(5*pi/4) q600[0];
cx q600[0],q600[1];
rx(pi/4) q600[1];
