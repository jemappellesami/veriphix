OPENQASM 2.0;
include "qelib1.inc";
qreg q577[6];
rz(3*pi/4) q577[2];
cx q577[3],q577[2];
cx q577[2],q577[1];
cx q577[0],q577[1];
rx(pi/4) q577[1];
