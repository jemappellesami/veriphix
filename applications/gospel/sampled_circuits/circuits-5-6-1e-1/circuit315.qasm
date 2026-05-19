OPENQASM 2.0;
include "qelib1.inc";
qreg q316[5];
cx q316[4],q316[3];
cx q316[3],q316[2];
cx q316[1],q316[2];
cx q316[1],q316[0];
rx(pi/4) q316[1];
