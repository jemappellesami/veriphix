OPENQASM 2.0;
include "qelib1.inc";
qreg q559[5];
cx q559[1],q559[2];
rz(pi/4) q559[2];
cx q559[2],q559[3];
cx q559[1],q559[2];
cx q559[1],q559[0];
