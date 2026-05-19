OPENQASM 2.0;
include "qelib1.inc";
qreg q785[4];
rx(3*pi/4) q785[0];
cx q785[3],q785[2];
rz(pi/2) q785[2];
cx q785[2],q785[1];
cx q785[1],q785[0];
