OPENQASM 2.0;
include "qelib1.inc";
qreg q963[4];
rx(7*pi/4) q963[2];
cx q963[2],q963[3];
cx q963[1],q963[2];
cx q963[0],q963[1];
rx(pi/4) q963[1];
