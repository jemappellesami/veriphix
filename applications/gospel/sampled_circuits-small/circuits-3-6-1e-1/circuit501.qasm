OPENQASM 2.0;
include "qelib1.inc";
qreg q502[3];
rx(pi/4) q502[2];
cx q502[1],q502[2];
cx q502[0],q502[1];
rx(pi/4) q502[1];
