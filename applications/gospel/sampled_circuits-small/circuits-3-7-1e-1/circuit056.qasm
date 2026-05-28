OPENQASM 2.0;
include "qelib1.inc";
qreg q57[3];
rx(pi/2) q57[2];
rz(pi/4) q57[2];
rx(pi) q57[2];
cx q57[2],q57[1];
cx q57[1],q57[0];
