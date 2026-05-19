OPENQASM 2.0;
include "qelib1.inc";
qreg q893[4];
cx q893[0],q893[1];
cx q893[3],q893[2];
cx q893[2],q893[1];
rx(pi/4) q893[0];
cx q893[0],q893[1];
