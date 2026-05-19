OPENQASM 2.0;
include "qelib1.inc";
qreg q662[5];
cx q662[4],q662[3];
cx q662[3],q662[2];
cx q662[1],q662[2];
cx q662[1],q662[0];
rx(pi/4) q662[1];
