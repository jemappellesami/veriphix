OPENQASM 2.0;
include "qelib1.inc";
qreg q536[3];
cx q536[0],q536[1];
rx(pi/4) q536[1];
rx(3*pi/2) q536[0];
cx q536[2],q536[1];
cx q536[0],q536[1];
