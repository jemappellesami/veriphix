OPENQASM 2.0;
include "qelib1.inc";
qreg q918[4];
rx(pi/4) q918[3];
cx q918[2],q918[3];
cx q918[1],q918[2];
cx q918[0],q918[1];
