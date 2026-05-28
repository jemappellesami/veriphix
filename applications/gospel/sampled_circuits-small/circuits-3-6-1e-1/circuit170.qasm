OPENQASM 2.0;
include "qelib1.inc";
qreg q171[3];
cx q171[1],q171[2];
rz(pi) q171[1];
rx(3*pi/2) q171[1];
cx q171[0],q171[1];
rx(pi/4) q171[1];
