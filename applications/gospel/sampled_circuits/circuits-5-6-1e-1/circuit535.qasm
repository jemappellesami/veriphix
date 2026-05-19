OPENQASM 2.0;
include "qelib1.inc";
qreg q536[5];
cx q536[4],q536[3];
cx q536[3],q536[2];
cx q536[2],q536[1];
cx q536[1],q536[0];
rx(pi/4) q536[1];
