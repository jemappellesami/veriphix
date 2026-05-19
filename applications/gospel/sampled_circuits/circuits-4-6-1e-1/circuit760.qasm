OPENQASM 2.0;
include "qelib1.inc";
qreg q761[4];
rz(pi/4) q761[3];
cx q761[2],q761[3];
cx q761[2],q761[1];
cx q761[0],q761[1];
rx(pi/4) q761[1];
