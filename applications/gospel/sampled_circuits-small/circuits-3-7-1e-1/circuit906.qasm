OPENQASM 2.0;
include "qelib1.inc";
qreg q907[3];
rx(pi/4) q907[0];
cx q907[1],q907[0];
cx q907[0],q907[1];
cx q907[2],q907[1];
cx q907[1],q907[0];
