OPENQASM 2.0;
include "qelib1.inc";
qreg q989[5];
rz(3*pi/4) q989[3];
cx q989[2],q989[3];
cx q989[2],q989[1];
cx q989[1],q989[0];
