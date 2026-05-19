OPENQASM 2.0;
include "qelib1.inc";
qreg q170[3];
cx q170[0],q170[1];
rz(pi) q170[1];
rx(5*pi/4) q170[0];
cx q170[2],q170[1];
cx q170[0],q170[1];
