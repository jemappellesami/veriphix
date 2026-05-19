OPENQASM 2.0;
include "qelib1.inc";
qreg q281[7];
cx q281[4],q281[5];
cx q281[3],q281[4];
cx q281[2],q281[3];
cx q281[1],q281[2];
cx q281[1],q281[0];
