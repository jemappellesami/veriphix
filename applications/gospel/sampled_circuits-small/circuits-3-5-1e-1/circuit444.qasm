OPENQASM 2.0;
include "qelib1.inc";
qreg q445[3];
cx q445[1],q445[0];
rz(pi/2) q445[2];
rx(7*pi/4) q445[2];
cx q445[1],q445[2];
cx q445[1],q445[0];
