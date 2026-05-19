OPENQASM 2.0;
include "qelib1.inc";
qreg q558[5];
cx q558[0],q558[1];
cx q558[2],q558[3];
cx q558[1],q558[0];
cx q558[2],q558[1];
cx q558[1],q558[0];
