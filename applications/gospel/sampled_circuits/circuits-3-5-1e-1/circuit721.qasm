OPENQASM 2.0;
include "qelib1.inc";
qreg q722[3];
cx q722[0],q722[1];
cx q722[1],q722[0];
cx q722[2],q722[1];
cx q722[1],q722[0];
