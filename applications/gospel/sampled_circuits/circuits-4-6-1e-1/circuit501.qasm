OPENQASM 2.0;
include "qelib1.inc";
qreg q502[4];
rx(5*pi/4) q502[3];
cx q502[2],q502[3];
cx q502[1],q502[2];
cx q502[0],q502[1];
rx(pi/4) q502[1];
