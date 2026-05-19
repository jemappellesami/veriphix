OPENQASM 2.0;
include "qelib1.inc";
qreg q469[7];
cx q469[3],q469[4];
cx q469[2],q469[3];
cx q469[2],q469[1];
cx q469[0],q469[1];
rx(pi/4) q469[1];
